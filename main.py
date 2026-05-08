from fastapi import FastAPI
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

model = SentenceTransformer('all-MiniLM-L6-v2')

def normalizar_texto(texto):
  if texto is None:
    return ""

  return str(texto).lower().strip()

def calcular_match_requisitos(texto_candidato, requisitos):
  if not requisitos:
    return 0
  
  texto_candidato = normalizar_texto(texto_candidato)
  matches = 0

  for requisito in requisitos:
    requisito = normalizar_texto(requisito)

    if requisito in texto_candidato:
      matches += 1

  return matches / len(requisitos)

@app.post("/ranking")
async def ranking(data:dict):
  
  vaga = data.get("vaga", {})
  candidatos = data.get("candidatos", [])

  nome_vaga = vaga.get(
    "nome_vaga", ""
  )

  descricao = vaga.get(
    "descricao", ""
  )

  requisitos_obrigatorios = vaga.get(
    "requisitos_obrigatorios", []
  )

  requisitos_desejaveis = vaga.get(
    "requisitos_desejaveis", []
  )

  texto_vaga = f"""
  Nome da vaga: {nome_vaga}
  Descricao: {descricao}
  Requisitos obrigatorios: {",".join(requisitos_obrigatorios)}
  Requisitos desejaveis: {",".join(requisitos_desejaveis)}
  """

  embedding_vaga = model.encode(model.encode(texto_vaga), convert_to_tensor=True)

  resultados = []

  for candidato in candidatos:
    texto_candidato = f"""
    Tecnico: {candidato.get('Tecnico', '')}
    Superior: {candidato.get('Superior', '')}
    Experiencias: {candidato.get('Experiencias', '')}
    Ferramentas: {candidato.get('Ferramentas', '')}
    Contribuicoes: {candidato.get('Contribuicoes', '')}
    """

    embedding_candidato = model.encode(texto_candidato, convert_to_tensor=True)

    score_semantico = util.cos_sim(embedding_vaga, embedding_candidato).item()

    score_obrigatorio = (calcular_match_requisitos(texto_candidato, requisitos_obrigatorios))

    score_desejaveis = (calcular_match_requisitos(texto_candidato, requisitos_desejaveis))

    score_final = (score_semantico * 0.6) + (score_obrigatorio * 0.3) + (score_desejaveis * 0,1)

    eliminado = False
    if requisitos_obrigatorios:
      if score_obrigatorio < 0.5:
        eliminado = True

    resultados.append({
      "Nome":
      candidato.get("Nome", ""),

      "Email":
      candidato.get("Email", ""),

      "BairroCidade":
      candidato.get(
          "BairroCidade",
          ""
      ),

      "Tecnico":
      candidato.get(
          "Tecnico",
          ""
      ),

      "Superior":
      candidato.get(
          "Superior",
          ""
      ),

      "PossuiParente":
      candidato.get(
          "PossuiParente",
          ""
      ),

      "NomeParente":
      candidato.get(
          "NomeParente",
          ""
      ),

      "PCD":
      candidato.get(
          "PCD",
          ""
      ),

      "TipoPCD":
      candidato.get(
          "TipoPCD",
          ""
      ),
      "score_final":
      round(score_final * 100, 2),

      "score_semantico":
      round(
          score_semantico * 100,
          2
      ),

      "score_obrigatorios":
      round(
          score_obrigatorio * 100,
          2
      ),

      "score_desejaveis":
      round(
          score_desejaveis * 100,
          2
      ),

      "eliminado":
      eliminado
    })
    
  resultados = sorted(resultados, key=lambda x: x["score_final"], reverse=True)

  return {
    "vaga": nome_vaga,
    "ranking": resultados
  }
