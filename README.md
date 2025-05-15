# UniV3 LP Simulator Alpha version

📊 Um simulador baseado em Streamlit para estimar e comparar os rendimentos de Provedores de Liquidez (LP) na Uniswap V3 da rede base. Esta ferramenta auxilia LPs a analisar retornos potenciais ao fornecer liquidez em pools da Uniswap V3, considerando estratégias de liquidez concentrada e diversos ajustes do mundo real.

## ✨ Funcionalidades Principais

*   **Descoberta de Pools:** Busca as principais pools da Uniswap V3 base utilizando The Graph.
*   **Dados Detalhados da Pool:** Exibe TVL atual, volume, taxas, preços e dados históricos de APY.
*   **Simulação Comparativa:**
    *   **Estratégia V2 (Faixa Completa):** Simula os retornos para uma posição de LP tradicional de faixa completa (0 a infinito).
    *   **Estratégia V3 (Liquidez Concentrada):**
        *   Defina faixas de preço personalizadas (`Preço Mínimo`, `Preço Máximo`).
        *   Calcula o **Fator de Amplificação da Liquidez** em comparação com uma posição V2.
        *   Estima o **Tempo na Faixa (TiR)** com base na volatilidade histórica de 7 dias ou entrada manual.
        *   Aplica a **Eficiência de Captura de Taxas**, suportando:
            *   **Entrada Manual (Slider):** Porcentagem de eficiência definida pelo usuário.
            *   **Baseada em Dados (Beta):** Calcula a eficiência com base na sua participação na liquidez ativa no tick atual da pool (requer o carregamento de todos os ticks ativos).
        *   Fornece **APR e Taxas V3 Ajustadas** após considerar TiR e eficiência de captura.
*   **Visualização da Distribuição de Liquidez:**
    *   Gráfico interativo (Plotly) mostrando a distribuição da `liquidityGross` através de diferentes ticks de preço para a pool selecionada.
    *   Ajuda a visualizar onde outros LPs concentraram sua liquidez.
*   **Interface Amigável:** Construído com Streamlit para fácil interação e ajuste de parâmetros.
*   **Insights Baseados em Dados:** Utiliza dados em tempo real e históricos do The Graph.
*   **Transparência nos Cálculos:** Fornece notas sobre os cálculos V3 se certas condições levarem a resultados zero ou inesperados (ex: fora da faixa, amplificação zero).

## 🚀 Como Começar

### Pré-requisitos

*   Python 3.8+
*   PIP (instalador de pacotes Python)

### Instalação

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/SEU_USUARIO/SEU_PROJETO.git
    cd SEU_PROJETO
    ```

2.  **Crie e ative um ambiente virtual (recomendado):**
    ```bash
    python -m venv .venv
    # No Windows
    .venv\Scripts\activate
    # No macOS/Linux
    source .venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Você precisará criar um arquivo `requirements.txt`. Veja a seção abaixo)*

4.  **Configure sua Chave de API do The Graph:**
    *   Abra o script Python (ex: `app.py` ou `test_2.py`).
    *   Localize a linha: `GRAPH_API_KEY = 'SUA_CHAVE_DE_API_AQUI'`
    *   Substitua `'SUA_CHAVE_DE_API_AQUI'` pela sua chave de API real do The Graph.
    *   **Nota de Segurança:** Para aplicações em produção, é altamente recomendável usar os "Secrets" do Streamlit ou variáveis de ambiente para gerenciar sua chave de API em vez de codificá-la diretamente.

### Executando a Aplicação

```bash
streamlit run seu_script.py
