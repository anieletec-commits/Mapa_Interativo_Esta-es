from pathlib import Path
import csv
import re
import unicodedata
import webbrowser

import numpy as np
import pandas as pd
import geopandas as gpd
import folium

from folium.plugins import Fullscreen, MiniMap, MousePosition

# 1. CAMINHOS
# ============================================================

PASTA_DADOS = Path(
    r"C:\Users\ResTIC55\Desktop\A.A\UFPEL-UFMG\ORGANIZAÇÃO_PROCESSADA\DADOS_DENTRO_DAS_BACIAS"
)

CAMINHO_SHAPE = Path(
    r"C:\Users\ResTIC55\Desktop\A.A\UFPEL-UFMG"
    r"\shapefiles_bacias\Bacias_Combinadas_RS.shp"
)

PASTA_SAIDA = Path(
    r"C:\Users\ResTIC55\Desktop\A.A\UFPEL-UFMG"
    r"\MAPA_BACIAS_ESTACOES"
)

ARQUIVO_HTML = (
    PASTA_SAIDA
    / "Mapa_Interativo_Bacias_Estacoes.html"
)

ARQUIVO_CSV = (
    PASTA_SAIDA
    / "Estacoes_usadas_no_mapa.csv"
)

ARQUIVO_ERROS = (
    PASTA_SAIDA
    / "Arquivos_com_erro.csv"
)


# 2. CONFIGURAÇÕES
# ============================================================

LIMITE_MEDIA = 50.0

ABRIR_MAPA_AUTOMATICAMENTE = True

FILTRAR_DENTRO_DAS_BACIAS = True

TAMANHO_BLOCO = 200_000

EXTENSOES_ACEITAS = {
    ".csv",
    ".txt",
}


# 3. POSIÇÃO DAS COLUNAS
# ============================================================
# A = Data
# B = Hora
# C = Valor
# D = Unidade
# E = Código
# F = Nome da estação
# G = Latitude
# H = Longitude
# I = Fonte

INDICE_VALOR = 2
INDICE_UNIDADE = 3
INDICE_CODIGO = 4
INDICE_NOME = 5
INDICE_LATITUDE = 6
INDICE_LONGITUDE = 7
INDICE_FONTE = 8


# 4. CORES
# ============================================================

CORES = {
    "Cota_clara": "#64B5F6",
    "Cota_escura": "#0D47A1",

    "Vazão_clara": "#EF9A9A",
    "Vazão_escura": "#B71C1C",

    "Precipitação": "#00897B",
    "Temperatura": "#EF6C00",
    "Outro": "#7B1FA2",
}

CORES_BACIAS = [
    "#90CAF9",
    "#A5D6A7",
    "#FFAB91",
    "#CE93D8",
    "#FFE082",
    "#80CBC4",
    "#BCAAA4",
    "#F48FB1",
]

# 5. FUNÇÕES 
# ============================================================

def normalizar_texto(texto):
    
    texto = str(texto).strip().lower()

    texto = unicodedata.normalize(
        "NFKD",
        texto,
    )

    texto = "".join(
        caractere
        for caractere in texto
        if not unicodedata.combining(caractere)
    )

    texto = re.sub(
        r"[^a-z0-9]+",
        "_",
        texto,
    )

    return texto.strip("_")


def converter_numero(valor):

    if valor is None:
        return np.nan

    texto = str(valor).strip()

    if texto.lower() in {
        "",
        "nan",
        "none",
        "null",
        "na",
        "n/a",
        "-",
        "--",
    }:
        return np.nan

    texto = texto.replace("\u00a0", "")
    texto = texto.replace("−", "-")
    texto = texto.replace(" ", "")

    if "," in texto and "." in texto:

        if texto.rfind(",") > texto.rfind("."):
            texto = texto.replace(".", "")
            texto = texto.replace(",", ".")

        else:
            texto = texto.replace(",", "")

    elif "," in texto:
        texto = texto.replace(",", ".")

    try:
        numero = float(texto)

    except ValueError:
        return np.nan

    if not np.isfinite(numero):
        return np.nan

    return numero


def converter_valor_media(valor):
    """
    Somente valores maiores que zero entram na média de
    cota e vazão.
    """
    numero = converter_numero(valor)

    if pd.isna(numero):
        return np.nan

    if numero <= 0:
        return np.nan

    if abs(numero) > 1e30:
        return np.nan

    return numero


def limpar_codigo(valor):
    """
    Remove '.0' de códigos que foram interpretados como número.
    """
    texto = str(valor).strip()

    if texto.lower() in {
        "",
        "nan",
        "none",
        "null",
    }:
        return ""

    return re.sub(
        r"\.0$",
        "",
        texto,
    )


def limpar_texto(valor):
    texto = str(valor).strip()

    if texto.lower() in {
        "nan",
        "none",
        "null",
    }:
        return ""

    return texto

# 6. IDENTIFICAÇÃO DO TIPO DE DADO
# ============================================================

def identificar_tipo(arquivo):
    """
    Identifica o tipo pelo nome do arquivo ou da pasta.
    """
    texto = normalizar_texto(
        str(arquivo)
    )

    if any(
        palavra in texto
        for palavra in [
            "cota",
            "cotas",
            "stage",
            "nivel",
        ]
    ):
        return "Cota"

    if any(
        palavra in texto
        for palavra in [
            "vazao",
            "vazoes",
            "flow",
            "discharge",
        ]
    ):
        return "Vazão"

    if any(
        palavra in texto
        for palavra in [
            "precipitacao",
            "precipitacoes",
            "precip",
            "chuva",
            "rain",
        ]
    ):
        return "Precipitação"

    if any(
        palavra in texto
        for palavra in [
            "temperatura",
            "temperaturas",
            "temperature",
            "temp",
        ]
    ):
        return "Temperatura"

    return "Outro"

# 7. LEITURA DOS ARQUIVOS
# ============================================================

def detectar_separador(arquivo):
    """
    Detecta se o arquivo usa:
    ; , tabulação ou |
    """
    codificacoes = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1",
    ]

    amostra = ""

    for codificacao in codificacoes:
        try:
            with open(
                arquivo,
                "r",
                encoding=codificacao,
                errors="replace",
            ) as arquivo_aberto:
                amostra = arquivo_aberto.read(20000)

            if amostra.strip():
                break

        except Exception:
            continue

    if not amostra.strip():
        return ";"

    try:
        return csv.Sniffer().sniff(
            amostra,
            delimiters=";,\t|",
        ).delimiter

    except csv.Error:
        primeira_linha = amostra.splitlines()[0]

        contagens = {
            ";": primeira_linha.count(";"),
            ",": primeira_linha.count(","),
            "\t": primeira_linha.count("\t"),
            "|": primeira_linha.count("|"),
        }

        return max(
            contagens,
            key=contagens.get,
        )


def detectar_codificacao(
    arquivo,
    separador,
):
    """
    Testa as codificações mais comuns.
    """
    codificacoes = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1",
    ]

    for codificacao in codificacoes:
        try:
            pd.read_csv(
                arquivo,
                sep=separador,
                encoding=codificacao,
                dtype=str,
                nrows=5,
                keep_default_na=False,
            )

            return codificacao

        except Exception:
            continue

    raise ValueError(
        f"Não foi possível ler o arquivo: {arquivo.name}"
    )


def localizar_coluna(
    colunas,
    nomes_possiveis,
    indice_reserva,
):
    """
    Procura a coluna pelo nome.

    Caso não encontre, usa a posição informada.
    """
    nomes_normalizados = {
        normalizar_texto(nome)
        for nome in nomes_possiveis
    }

    for coluna in colunas:
        nome_coluna = normalizar_texto(
            coluna
        )

        if nome_coluna in nomes_normalizados:
            return coluna

    if len(colunas) > indice_reserva:
        return colunas[indice_reserva]

    return None


def identificar_colunas(colunas):
    """
    Localiza as colunas necessárias.
    """
    resultado = {
        "valor": localizar_coluna(
            colunas,
            {
                "valor",
                "value",
                "cota",
                "vazao",
                "precipitacao",
                "temperatura",
            },
            INDICE_VALOR,
        ),

        "unidade": localizar_coluna(
            colunas,
            {
                "unidade",
                "unit",
            },
            INDICE_UNIDADE,
        ),

        "codigo": localizar_coluna(
            colunas,
            {
                "codigo",
                "codigo_estacao",
                "cod_estacao",
            },
            INDICE_CODIGO,
        ),

        "nome": localizar_coluna(
            colunas,
            {
                "nome",
                "nome_estacao",
                "estacao",
            },
            INDICE_NOME,
        ),

        "latitude": localizar_coluna(
            colunas,
            {
                "latitude",
                "lat",
            },
            INDICE_LATITUDE,
        ),

        "longitude": localizar_coluna(
            colunas,
            {
                "longitude",
                "lon",
                "long",
                "lng",
            },
            INDICE_LONGITUDE,
        ),

        "fonte": localizar_coluna(
            colunas,
            {
                "fonte",
                "source",
                "origem",
            },
            INDICE_FONTE,
        ),
    }

    obrigatorias = [
        "codigo",
        "nome",
        "latitude",
        "longitude",
    ]

    for coluna in obrigatorias:
        if resultado[coluna] is None:
            raise ValueError(
                f"Coluna '{coluna}' não encontrada."
            )

    return resultado

# 8. PROCESSAMENTO DE CADA ARQUIVO
# ============================================================

def processar_arquivo(arquivo):
    """
    Extrai as estações de um arquivo.

    Para cota e vazão, calcula a média dos valores maiores
    que zero.

    Para precipitação e temperatura, apenas identifica as
    estações e suas coordenadas.
    """
    tipo = identificar_tipo(arquivo)

    separador = detectar_separador(
        arquivo
    )

    codificacao = detectar_codificacao(
        arquivo,
        separador,
    )

    cabecalho = pd.read_csv(
        arquivo,
        sep=separador,
        encoding=codificacao,
        dtype=str,
        nrows=0,
    )

    colunas = identificar_colunas(
        list(cabecalho.columns)
    )

    acumuladores = {}

    leitor = pd.read_csv(
        arquivo,
        sep=separador,
        encoding=codificacao,
        dtype=str,
        chunksize=TAMANHO_BLOCO,
        keep_default_na=False,
        na_filter=False,
        low_memory=False,
    )

    for bloco in leitor:

        tabela = pd.DataFrame({
            "codigo": bloco[
                colunas["codigo"]
            ].map(limpar_codigo),

            "nome": bloco[
                colunas["nome"]
            ].map(limpar_texto),

            "latitude": bloco[
                colunas["latitude"]
            ].map(converter_numero),

            "longitude": bloco[
                colunas["longitude"]
            ].map(converter_numero),
        })

        if colunas["fonte"] is not None:
            tabela["fonte"] = bloco[
                colunas["fonte"]
            ].map(limpar_texto)
        else:
            tabela["fonte"] = ""

        if colunas["unidade"] is not None:
            tabela["unidade"] = bloco[
                colunas["unidade"]
            ].map(limpar_texto)
        else:
            tabela["unidade"] = ""

        # Média apenas para cota e vazão.
        if (
            tipo in {"Cota", "Vazão"}
            and colunas["valor"] is not None
        ):
            tabela["valor"] = bloco[
                colunas["valor"]
            ].map(converter_valor_media)

        else:
            tabela["valor"] = np.nan

        # Remove somente coordenadas inválidas.
        tabela = tabela[
            tabela["latitude"].between(
                -90,
                90,
            )
            & tabela["longitude"].between(
                -180,
                180,
            )
        ].copy()

        if tabela.empty:
            continue

        tabela["tipo"] = tipo

        agrupado = (
            tabela.groupby(
                [
                    "codigo",
                    "nome",
                    "latitude",
                    "longitude",
                    "fonte",
                    "unidade",
                    "tipo",
                ],
                dropna=False,
                as_index=False,
            )
            .agg(
                soma=("valor", "sum"),
                quantidade=("valor", "count"),
            )
        )

        for _, linha in agrupado.iterrows():

            chave = (
                linha["codigo"],
                linha["nome"],
                round(
                    float(linha["latitude"]),
                    7,
                ),
                round(
                    float(linha["longitude"]),
                    7,
                ),
                linha["fonte"],
                linha["unidade"],
                linha["tipo"],
            )

            if chave not in acumuladores:
                acumuladores[chave] = {
                    "codigo": linha["codigo"],
                    "nome": linha["nome"],
                    "latitude": linha["latitude"],
                    "longitude": linha["longitude"],
                    "fonte": linha["fonte"],
                    "unidade": linha["unidade"],
                    "tipo": linha["tipo"],
                    "soma": 0.0,
                    "quantidade": 0,
                }

            acumuladores[chave]["soma"] += float(
                linha["soma"]
            )

            acumuladores[chave]["quantidade"] += int(
                linha["quantidade"]
            )

    resultados = []

    for registro in acumuladores.values():

        if registro["quantidade"] > 0:
            media = (
                registro["soma"]
                / registro["quantidade"]
            )
        else:
            media = np.nan

        resultados.append({
            "codigo": registro["codigo"],
            "nome": registro["nome"],
            "latitude": registro["latitude"],
            "longitude": registro["longitude"],
            "fonte": registro["fonte"],
            "unidade": registro["unidade"],
            "tipo": registro["tipo"],
            "media": media,
            "quantidade_valores_media": registro[
                "quantidade"
            ],
            "arquivo_origem": arquivo.name,
        })

    return pd.DataFrame(
        resultados
    )


# 9. TODOS OS ARQUIVOS
# ============================================================

def coletar_estacoes():
    """
    Processa todos os arquivos da pasta.
    """
    arquivos = sorted(
        arquivo
        for arquivo in PASTA_DADOS.rglob("*")
        if (
            arquivo.is_file()
            and arquivo.suffix.lower()
            in EXTENSOES_ACEITAS
        )
    )

    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum arquivo encontrado em:\n"
            f"{PASTA_DADOS}"
        )

    tabelas = []
    erros = []

    print(
        f"Arquivos encontrados: {len(arquivos)}"
    )
    print()

    for numero, arquivo in enumerate(
        arquivos,
        start=1,
    ):
        print(
            f"[{numero}/{len(arquivos)}] "
            f"{arquivo.name}"
        )

        try:
            tabela = processar_arquivo(
                arquivo
            )

            if tabela.empty:
                print(
                    "  Nenhuma estação válida."
                )

            else:
                tabelas.append(tabela)

                print(
                    f"  Estações encontradas: "
                    f"{len(tabela)}"
                )

        except Exception as erro:
            print(
                f"  ERRO: {erro}"
            )

            erros.append({
                "arquivo": str(arquivo),
                "erro": str(erro),
            })

    if not tabelas:
        raise ValueError(
            "Nenhuma estação válida foi encontrada."
        )

    estacoes = pd.concat(
        tabelas,
        ignore_index=True,
    )

    # Valor auxiliar para unir médias de arquivos diferentes.
    estacoes["soma_auxiliar"] = (
        estacoes["media"].fillna(0)
        * estacoes["quantidade_valores_media"]
    )

    estacoes = (
        estacoes.groupby(
            [
                "codigo",
                "nome",
                "latitude",
                "longitude",
                "fonte",
                "unidade",
                "tipo",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            soma_auxiliar=(
                "soma_auxiliar",
                "sum",
            ),

            quantidade_valores_media=(
                "quantidade_valores_media",
                "sum",
            ),

            arquivos_origem=(
                "arquivo_origem",
                lambda valores: " | ".join(
                    sorted(
                        set(
                            str(valor)
                            for valor in valores
                        )
                    )
                ),
            ),
        )
    )

    estacoes["media"] = np.where(
        estacoes[
            "quantidade_valores_media"
        ] > 0,
        (
            estacoes["soma_auxiliar"]
            / estacoes[
                "quantidade_valores_media"
            ]
        ),
        np.nan,
    )

    estacoes = estacoes.drop(
        columns="soma_auxiliar"
    )

    # Classificação usada nas cores.
    estacoes["classe"] = "Cor única"

    mascara_cota_vazao = (
        estacoes["tipo"].isin(
            ["Cota", "Vazão"]
        )
    )

    estacoes.loc[
        mascara_cota_vazao
        & (
            estacoes["media"]
            <= LIMITE_MEDIA
        ),
        "classe",
    ] = "Média até 50"

    estacoes.loc[
        mascara_cota_vazao
        & (
            estacoes["media"]
            > LIMITE_MEDIA
        ),
        "classe",
    ] = "Média acima de 50"

    estacoes.loc[
        mascara_cota_vazao
        & estacoes["media"].isna(),
        "classe",
    ] = "Sem média válida"

    return (
        estacoes,
        pd.DataFrame(erros),
    )

# 10. LEITURA DAS BACIAS
# ============================================================

def preparar_bacias():
    """
    Lê o shapefile e converte para EPSG:4326,
    usado pelo mapa interativo.
    """
    if not CAMINHO_SHAPE.exists():
        raise FileNotFoundError(
            f"Shapefile não encontrado:\n"
            f"{CAMINHO_SHAPE}"
        )

    bacias = gpd.read_file(
        CAMINHO_SHAPE
    )

    if bacias.empty:
        raise ValueError(
            "O shapefile está vazio."
        )

    if bacias.crs is None:
        raise ValueError(
            "O shapefile não possui CRS definido."
        )

    bacias = bacias[
        bacias.geometry.notna()
        & ~bacias.geometry.is_empty
    ].copy()

    # Corrige geometrias inválidas.
    invalidas = (
        ~bacias.geometry.is_valid
    )

    if invalidas.any():
        bacias.loc[
            invalidas,
            "geometry",
        ] = (
            bacias.loc[
                invalidas,
                "geometry",
            ]
            .geometry
            .buffer(0)
        )

    # Converte para latitude e longitude.
    bacias = bacias.to_crs(
        epsg=4326
    )

    # Tenta localizar uma coluna com o nome da bacia.
    coluna_nome = None

    for coluna in bacias.columns:

        if coluna == "geometry":
            continue

        nome_normalizado = normalizar_texto(
            coluna
        )

        if nome_normalizado in {
            "nome",
            "bacia",
            "nome_bacia",
            "name",
        }:
            coluna_nome = coluna
            break

    if coluna_nome is None:
        bacias["nome_mapa"] = [
            f"Bacia {numero + 1}"
            for numero in range(
                len(bacias)
            )
        ]

    else:
        bacias["nome_mapa"] = (
            bacias[coluna_nome]
            .astype(str)
            .str.replace(
                "_",
                " ",
                regex=False,
            )
            .str.strip()
        )

    return bacias

# 11. FILTRO ESPACIAL
# ============================================================

def filtrar_estacoes(
    estacoes,
    bacias,
):
    """
    Mantém somente estações que estiverem dentro ou sobre
    alguma bacia.
    """
    pontos = gpd.GeoDataFrame(
        estacoes.copy(),
        geometry=gpd.points_from_xy(
            estacoes["longitude"],
            estacoes["latitude"],
        ),
        crs="EPSG:4326",
    )

    if not FILTRAR_DENTRO_DAS_BACIAS:
        return pontos

    area_bacias = (
        bacias.geometry.union_all()
    )

    dentro = pontos.geometry.intersects(
        area_bacias
    )

    removidas = int(
        (~dentro).sum()
    )

    print()
    print(
        f"Estações fora das bacias: {removidas}"
    )

    return pontos.loc[
        dentro
    ].copy()

# 12. INFORMAÇÕES DOS PONTOS
# ============================================================

def formatar_media(valor):
    if pd.isna(valor):
        return "Não calculada"

    return f"{valor:.3f}".replace(
        ".",
        ",",
    )


def criar_popup(linha):
    """
    Conteúdo exibido quando a estação é clicada.
    """
    media = formatar_media(
        linha["media"]
    )

    codigo = (
        linha["codigo"]
        if str(linha["codigo"]).strip()
        else "Não informado"
    )

    nome = (
        linha["nome"]
        if str(linha["nome"]).strip()
        else "Não informado"
    )

    fonte = (
        linha["fonte"]
        if str(linha["fonte"]).strip()
        else "Não informada"
    )

    unidade = (
        linha["unidade"]
        if str(linha["unidade"]).strip()
        else "Não informada"
    )

    return f"""
    <div style="
        font-family: Arial;
        font-size: 13px;
        width: 260px;
    ">
        <h4 style="
            margin-top: 0;
            margin-bottom: 8px;
            color: #263238;
        ">
            {nome}
        </h4>

        <b>Código:</b> {codigo}<br>
        <b>Tipo:</b> {linha['tipo']}<br>
        <b>Fonte:</b> {fonte}<br>
        <b>Unidade:</b> {unidade}<br>
        <b>Média:</b> {media}<br>
        <b>Classificação:</b> {linha['classe']}<br>
        <b>Latitude:</b> {linha['latitude']:.6f}<br>
        <b>Longitude:</b> {linha['longitude']:.6f}<br>
        <b>Valores usados:</b>
        {int(linha['quantidade_valores_media'])}
    </div>
    """

# 13. CORES DOS PONTOS
# ============================================================

def obter_cor(linha):
    tipo = linha["tipo"]

    if tipo == "Cota":

        if (
            pd.notna(linha["media"])
            and linha["media"] > LIMITE_MEDIA
        ):
            return CORES["Cota_escura"]

        return CORES["Cota_clara"]

    if tipo == "Vazão":

        if (
            pd.notna(linha["media"])
            and linha["media"] > LIMITE_MEDIA
        ):
            return CORES["Vazão_escura"]

        return CORES["Vazão_clara"]

    if tipo == "Precipitação":
        return CORES["Precipitação"]

    if tipo == "Temperatura":
        return CORES["Temperatura"]

    return CORES["Outro"]


def obter_raio(linha):
    """
    Pontos acima de 50 ficam um pouco maiores.
    """
    if (
        linha["tipo"] in {"Cota", "Vazão"}
        and pd.notna(linha["media"])
        and linha["media"] > LIMITE_MEDIA
    ):
        return 7

    return 5

# 14. LEGENDA
# ============================================================

def adicionar_legenda(mapa):
    legenda = f"""
    <div style="
        position: fixed;
        bottom: 35px;
        left: 35px;
        z-index: 9999;
        background-color: white;
        border: 2px solid #777;
        border-radius: 6px;
        padding: 12px;
        font-family: Arial;
        font-size: 13px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.35);
    ">

        <div style="
            font-weight: bold;
            margin-bottom: 9px;
            font-size: 14px;
        ">
            Legenda
        </div>

        <div>
            <span style="
                display:inline-block;
                width:13px;
                height:13px;
                border-radius:50%;
                background:{CORES['Cota_clara']};
                margin-right:6px;
            "></span>
            Cota — média até 50
        </div>

        <div>
            <span style="
                display:inline-block;
                width:13px;
                height:13px;
                border-radius:50%;
                background:{CORES['Cota_escura']};
                margin-right:6px;
            "></span>
            Cota — média acima de 50
        </div>

        <div>
            <span style="
                display:inline-block;
                width:13px;
                height:13px;
                border-radius:50%;
                background:{CORES['Vazão_clara']};
                margin-right:6px;
            "></span>
            Vazão — média até 50
        </div>

        <div>
            <span style="
                display:inline-block;
                width:13px;
                height:13px;
                border-radius:50%;
                background:{CORES['Vazão_escura']};
                margin-right:6px;
            "></span>
            Vazão — média acima de 50
        </div>

        <div>
            <span style="
                display:inline-block;
                width:13px;
                height:13px;
                border-radius:50%;
                background:{CORES['Precipitação']};
                margin-right:6px;
            "></span>
            Precipitação
        </div>

        <div>
            <span style="
                display:inline-block;
                width:13px;
                height:13px;
                border-radius:50%;
                background:{CORES['Temperatura']};
                margin-right:6px;
            "></span>
            Temperatura
        </div>

    </div>
    """

    mapa.get_root().html.add_child(
        folium.Element(legenda)
    )

# 15. TÍTULO
# ============================================================

def adicionar_titulo(mapa):
    titulo = """
    <div style="
        position: fixed;
        top: 10px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 9999;
        background-color: rgba(255,255,255,0.94);
        border: 1px solid #777;
        border-radius: 6px;
        padding: 10px 20px;
        font-family: Arial;
        font-size: 18px;
        font-weight: bold;
        color: #263238;
        box-shadow: 0 1px 6px rgba(0,0,0,0.30);
        white-space: nowrap;
    ">
        Bacias hidrográficas e estações hidrometeorológicas
    </div>
    """

    mapa.get_root().html.add_child(
        folium.Element(titulo)
    )

# 16. CRIAÇÃO DO MAPA INTERATIVO
# ============================================================

def criar_mapa(
    bacias,
    estacoes,
):
    PASTA_SAIDA.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Centro inicial do mapa.
    centro = [
        float(
            bacias.geometry.centroid.y.mean()
        ),
        float(
            bacias.geometry.centroid.x.mean()
        ),
    ]

    mapa = folium.Map(
        location=centro,
        zoom_start=7,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    # Mapas-base.
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Ruas — OpenStreetMap",
        control=True,
        show=True,
    ).add_to(mapa)

    folium.TileLayer(
        tiles="CartoDB positron",
        name="Mapa claro",
        control=True,
        show=False,
    ).add_to(mapa)

    folium.TileLayer(
        tiles="CartoDB Voyager",
        name="Mapa detalhado",
        control=True,
        show=False,
    ).add_to(mapa)

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/World_Imagery/"
            "MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri World Imagery",
        name="Imagem de satélite",
        control=True,
        show=False,
    ).add_to(mapa)

    # Camada das bacias.
    grupo_bacias = folium.FeatureGroup(
        name="Bacias hidrográficas",
        show=True,
    )

    for numero, (_, bacia) in enumerate(
        bacias.iterrows()
    ):
        cor = CORES_BACIAS[
            numero % len(CORES_BACIAS)
        ]

        nome_bacia = str(
            bacia["nome_mapa"]
        )

        geometria = gpd.GeoSeries(
            [bacia.geometry],
            crs="EPSG:4326",
        ).__geo_interface__

        folium.GeoJson(
            data=geometria,
            style_function=lambda feature, cor=cor: {
                "fillColor": cor,
                "color": "#37474F",
                "weight": 2,
                "fillOpacity": 0.28,
            },
            highlight_function=lambda feature: {
                "weight": 4,
                "fillOpacity": 0.45,
            },
            tooltip=folium.Tooltip(
                f"Bacia: {nome_bacia}",
                sticky=True,
            ),
        ).add_to(grupo_bacias)

    grupo_bacias.add_to(mapa)

    # Uma camada para cada tipo de dado.
    grupos = {
        "Cota": folium.FeatureGroup(
            name="Estações de cota",
            show=True,
        ),

        "Vazão": folium.FeatureGroup(
            name="Estações de vazão",
            show=True,
        ),

        "Precipitação": folium.FeatureGroup(
            name="Estações de precipitação",
            show=True,
        ),

        "Temperatura": folium.FeatureGroup(
            name="Estações de temperatura",
            show=True,
        ),

        "Outro": folium.FeatureGroup(
            name="Outras estações",
            show=False,
        ),
    }

    for _, linha in estacoes.iterrows():

        tipo = linha["tipo"]

        if tipo not in grupos:
            tipo = "Outro"

        cor = obter_cor(
            linha
        )

        raio = obter_raio(
            linha
        )

        nome_estacao = (
            linha["nome"]
            if str(linha["nome"]).strip()
            else linha["codigo"]
        )

        folium.CircleMarker(
            location=[
                linha["latitude"],
                linha["longitude"],
            ],
            radius=raio,
            color="white",
            weight=1.5,
            fill=True,
            fill_color=cor,
            fill_opacity=0.92,
            tooltip=folium.Tooltip(
                (
                    f"{nome_estacao}<br>"
                    f"{linha['tipo']} — "
                    f"{linha['classe']}"
                ),
                sticky=True,
            ),
            popup=folium.Popup(
                criar_popup(linha),
                max_width=320,
            ),
        ).add_to(
            grupos[tipo]
        )

    for grupo in grupos.values():
        grupo.add_to(mapa)

    # Ajusta o zoom para mostrar todas as bacias.
    limites = bacias.total_bounds

    mapa.fit_bounds([
        [
            limites[1],
            limites[0],
        ],
        [
            limites[3],
            limites[2],
        ],
    ])

    # Ferramentas interativas.
    folium.LayerControl(
        collapsed=False,
        position="topright",
    ).add_to(mapa)

    Fullscreen(
        position="topleft",
        title="Tela cheia",
        title_cancel="Sair da tela cheia",
        force_separate_button=True,
    ).add_to(mapa)

    MiniMap(
        toggle_display=True,
        position="bottomright",
    ).add_to(mapa)

    MousePosition(
        position="bottomright",
        separator=" | ",
        prefix="Coordenadas:",
        num_digits=5,
    ).add_to(mapa)

    adicionar_titulo(
        mapa
    )

    adicionar_legenda(
        mapa
    )

    mapa.save(
        ARQUIVO_HTML
    )

    # Salva a tabela das estações efetivamente usadas.
    tabela_saida = pd.DataFrame(
        estacoes.drop(
            columns="geometry",
            errors="ignore",
        )
    )

    tabela_saida.to_csv(
        ARQUIVO_CSV,
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 70)
    print("MAPA INTERATIVO CRIADO")
    print("=" * 70)
    print(
        f"Estações diferentes: "
        f"{estacoes['codigo'].nunique()}"
    )
    print(
        f"Registros estação/tipo: "
        f"{len(estacoes)}"
    )
    print(
        f"HTML: {ARQUIVO_HTML}"
    )
    print(
        f"CSV:  {ARQUIVO_CSV}"
    )

    if ABRIR_MAPA_AUTOMATICAMENTE:
        webbrowser.open(
            ARQUIVO_HTML.resolve().as_uri()
        )

# 17. EXECUÇÃO
# ============================================================

def executar():
    print("=" * 70)
    print("MAPA INTERATIVO DE BACIAS E ESTAÇÕES")
    print("=" * 70)
    print(
        f"Pasta dos dados: {PASTA_DADOS}"
    )
    print(
        f"Shapefile:       {CAMINHO_SHAPE}"
    )
    print(
        f"Pasta de saída:  {PASTA_SAIDA}"
    )
    print()

    if not PASTA_DADOS.exists():
        raise FileNotFoundError(
            f"A pasta de dados não existe:\n"
            f"{PASTA_DADOS}"
        )

    PASTA_SAIDA.mkdir(
        parents=True,
        exist_ok=True,
    )

    # lê o shapefile e reprojeta as bacias para latitude e longitude.
    bacias = preparar_bacias()

    print(
        f"Bacias carregadas: {len(bacias)}"
    )

    estacoes, erros = coletar_estacoes()

    pontos = filtrar_estacoes(
        estacoes,
        bacias,
    )

    if pontos.empty:
        raise ValueError(
            "Nenhuma estação ficou dentro das bacias."
        )

    if not erros.empty:
        erros.to_csv(
            ARQUIVO_ERROS,
            sep=";",
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"Arquivos com erro: {len(erros)}"
        )
        print(
            f"Relatório: {ARQUIVO_ERROS}"
        )

    criar_mapa(
        bacias=bacias,
        estacoes=pontos,
    )


if __name__ == "__main__":
    executar()