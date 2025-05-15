# -*- BIBLIOTECAS NECESSÁRIAS -*-
import streamlit as st
import numpy as np
import math
import requests
from datetime import datetime, timedelta
import pandas as pd
import traceback
from scipy.stats import norm
import json
import logging  # Added logging
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
import math

try:
    import plotly.graph_objects as go  # IMPORTAÇÃO CONDICIONAL PLOTLY PARA DISTIBUIÇÃO DE LIQUIDEZ

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# --- Logging Setup ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)  # Set to DEBUG for more verbose local debugging
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False

# --- CONFIGURAÇÕES E QUERYS ---
GRAPH_API_KEY = ''  # Sua chave da API aqui
SUBGRAPH_URL = f"https://gateway.thegraph.com/api/{GRAPH_API_KEY}/subgraphs/id/BHWNsedAHtmTCzXxCCDfhPmm6iN9rxUhoRHdHKyujic3"

# -- REQUISITA PISCINAS DE LIQUIDEZ DA BASE
TOP_POOLS_QUERY = """
query GetTopPools($orderBy: String!, $orderDirection: String!, $first: Int!) {
  pools(first: $first, orderBy: $orderBy, orderDirection: $orderDirection, where: {totalValueLockedUSD_gt: 10000, txCount_gt: 100}) {
    id
    token0 { symbol id }
    token1 { symbol id }
    feeTier
    volumeUSD
    totalValueLockedUSD
  }
}
"""

# -- REQUISITA INFORMAÇÕES ADICIONAIS DA PISCINA
POOL_DETAILS_QUERY = """
query GetPoolDetails($poolId: ID!, $daysToFetch: Int!) {
  pool(id: $poolId) {
    id
    token0 { symbol id decimals }
    token1 { symbol id decimals }
    feeTier
    tick # Current tick of the pool
    totalValueLockedUSD
    token0Price # P1 / P0
    token1Price # P0 / P1 (Usaremos este como preço relativo base)
  }
  poolDayDatas(
    first: $daysToFetch
    orderBy: date
    orderDirection: desc
    where: {pool: $poolId}
  ) {
    date # Timestamp UNIX
    tvlUSD
    volumeUSD # Volume naquele dia
    feesUSD # Taxas naquele dia
    token1Price # Preço relativo P0/P1 naquele dia
  }
}
"""

# -- REQUISITA DADOS RELATIVOS A LIQUIDEZ DA PISCINA
ACTIVE_TICKS_QUERY = """
query GetActiveTicks($poolAddress: String!, $skip: Int!) {
  ticks(first: 1000, skip: $skip, orderBy: tickIdx, orderDirection: asc, where: { poolAddress: $poolAddress, liquidityNet_not: "0" }) {
    tickIdx
    liquidityNet
  }
}
"""


# --- Funções Auxiliares ---
def send_graphql_query(url, query, variables=None):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {GRAPH_API_KEY}'
    }
    payload = {'query': query}
    if variables: payload['variables'] = variables
    logger.debug(f"Sending GraphQL query to {url} with variables: {variables}")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        json_response = response.json()
        if 'errors' in json_response:
            logger.error(f"Erro GraphQL API ({url}): {json_response['errors']}")
            first_error_message = json_response['errors'][0].get('message', 'Erro desconhecido do Subgraph.')
            if 'auth error' in first_error_message.lower() or 'permission denied' in first_error_message.lower() or 'api key' in first_error_message.lower():
                st.error(
                    f"Erro de Autenticação/Permissão no Subgraph: {first_error_message}. Verifique a GRAPH_API_KEY e suas permissões para este subgraph.")
            else:
                st.warning(f"Erro na query do Subgraph: {first_error_message}")
            return None
        if 'data' not in json_response:
            logger.warning(f"Resposta GraphQL sem 'data' de {url}. Resposta: {json_response}")
            st.warning("Resposta inesperada do Subgraph (sem campo 'data').")
            return None
        return json_response
    except requests.exceptions.Timeout:
        logger.error(f"Erro GraphQL Request: Timeout ({url})")
        st.error("Erro de Rede: Timeout ao conectar ao Subgraph.")
        return None
    except requests.exceptions.RequestException as e:
        error_detail = ""
        status_code = e.response.status_code if e.response is not None else "N/A"
        try:
            error_detail = f" Response Text: {e.response.text}" if e.response else ""
        except:
            pass
        logger.error(f"Erro GraphQL Request: Status {status_code} - {e} ({url}){error_detail}")
        if status_code in [401, 403]:
            st.error(
                f"Erro de Autorização ({status_code}). Verifique se a GRAPH_API_KEY é válida e tem permissão para acessar este subgraph ID.")
        else:
            st.error(f"Erro de Rede ({status_code}) ao conectar ao Subgraph: {e}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Erro JSON Decode: {response.text} ({url})")
        st.error("Erro ao processar resposta do Subgraph (JSON inválido).")
        return None
    except Exception as e:
        logger.error(f"Erro inesperado GraphQL: {e}\n{traceback.format_exc()} ({url})")
        st.error("Erro inesperado na comunicação com o Subgraph.")
        return None


@st.cache_data(ttl=900)
def get_top_pools_subgraph(subgraph_url):
    logger.info(f"Buscando lista de top pools de {subgraph_url}")
    variables = {"orderBy": "totalValueLockedUSD", "orderDirection": "desc", "first": 100}
    data = send_graphql_query(subgraph_url, TOP_POOLS_QUERY, variables)
    pool_options = {}
    if data and 'data' in data and data['data'] and 'pools' in data['data']:
        pools_raw = data['data']['pools']
        if pools_raw is None: return {}
        for pool in pools_raw:
            try:
                pool_id = pool.get('id');
                fee_tier_bps = int(pool.get('feeTier', 0))
                t0_sym = pool.get('token0', {}).get('symbol', 'T0');
                t1_sym = pool.get('token1', {}).get('symbol', 'T1')
                tvl_last = float(pool.get('totalValueLockedUSD', 0))
                if not pool_id or fee_tier_bps <= 0 or t0_sym == 'T0' or t1_sym == 'T1': continue
                fee_tier_percent = fee_tier_bps / 10000.0
                display_name = f"{t0_sym}/{t1_sym} {fee_tier_percent:.2f}% (TVL: ${tvl_last:,.0f})"
                pool_options[pool_id] = {'display_name': display_name, 'token0_symbol': t0_sym, 'token1_symbol': t1_sym,
                                         'fee_tier_percent': fee_tier_percent, 'tvl_usd': tvl_last}
            except Exception as e:
                logger.warning(f"Erro processando pool básica {pool.get('id')}: {e}")
        sorted_pool_keys = sorted(pool_options, key=lambda k: pool_options[k]['tvl_usd'], reverse=True)
        return {k: pool_options[k] for k in sorted_pool_keys}
    return {}


@st.cache_data(ttl=180)
def get_detailed_pool_data(pool_id, subgraph_url, history_days=35):
    logger.info(f"Buscando dados detalhados para pool {pool_id} de {subgraph_url}...")
    variables = {"poolId": pool_id.lower(), "daysToFetch": history_days}
    data = send_graphql_query(subgraph_url, POOL_DETAILS_QUERY, variables)
    if not data or 'data' not in data: return None
    pool_data = data['data'].get('pool');
    day_datas_raw = data['data'].get('poolDayDatas')
    if pool_data is None: return None

    token0_data = pool_data.get('token0', {})
    token1_data = pool_data.get('token1', {})
    token0_decimals_val = int(token0_data.get('decimals', 18))
    token1_decimals_val = int(token1_data.get('decimals', 18))
    current_tick_val = int(pool_data.get('tick')) if pool_data.get('tick') is not None else None

    latest_tvl = float(pool_data.get('totalValueLockedUSD', 0.0))
    try:
        latest_price_p0_p1 = float(pool_data.get('token1Price', 0.0))
    except (TypeError, ValueError):
        latest_price_p0_p1 = 0.0
    try:
        latest_price_p1_p0 = float(pool_data.get('token0Price', 0.0))
    except (TypeError, ValueError):
        latest_price_p1_p0 = 0.0
    fee_tier_percent = float(pool_data.get('feeTier', 0)) / 10000.0
    token0_symbol = token0_data.get('symbol', 'T0');
    token1_symbol = token1_data.get('symbol', 'T1')
    token0_id = token0_data.get('id');
    token1_id = token1_data.get('id')

    if day_datas_raw is None or not isinstance(day_datas_raw, list): day_datas_raw = []
    base_return = {
        "pool_id": pool_id, "token0_symbol": token0_symbol, "token1_symbol": token1_symbol,
        "token0_id": token0_id, "token1_id": token1_id, "feeTier": fee_tier_percent,
        "token0_decimals": token0_decimals_val, "token1_decimals": token1_decimals_val,
        "current_tick": current_tick_val,
        "current_tvl_usd": latest_tvl,
        "current_price_p0_p1": latest_price_p0_p1 if latest_price_p0_p1 > 0 else None,
        "current_price_p1_p0": latest_price_p1_p0 if latest_price_p1_p0 > 0 else None,
        "volume_24h_approx_usd": 0.0, "apys": {'latest': 0.0, 'avg_7d': 0.0, 'avg_30d': 0.0},
        "volatility_7d": None, "volatility_30d": None,
        "last_update_subgraph": "N/A", "warning": None
    }
    if not day_datas_raw:
        base_return["warning"] = "Sem dados históricos."
        return base_return

    df_day = pd.DataFrame(day_datas_raw);
    last_update_timestamp = "N/A";
    volume_24h = 0.0
    apys = {'latest': 0.0, 'avg_7d': 0.0, 'avg_30d': 0.0};
    volatilities = {'vol_7d': None, 'vol_30d': None}
    processing_warning = None
    try:
        df_day['date_dt'] = pd.to_datetime(df_day['date'], unit='s', errors='coerce')
        df_day = df_day.dropna(subset=['date_dt']).set_index('date_dt').sort_index()
        numeric_cols = ['tvlUSD', 'volumeUSD', 'feesUSD', 'token1Price']
        for col in numeric_cols: df_day[col] = pd.to_numeric(df_day[col], errors='coerce')
        df_day = df_day.dropna(subset=['tvlUSD', 'token1Price'])
        df_day = df_day[(df_day['tvlUSD'] > 1) & (df_day['token1Price'] > 1e-18)]
        df_day['feesUSD'] = df_day['feesUSD'].fillna(0.0);
        df_day['volumeUSD'] = df_day['volumeUSD'].fillna(0.0)
        if not df_day.empty:
            last_update_timestamp = df_day.index.max().strftime("%Y-%m-%d %H:%M UTC")
            volume_24h = df_day.iloc[-1]['volumeUSD']
            end_date = df_day.index.max();
            last_day_data = df_day.iloc[-1]
            if last_day_data['tvlUSD'] > 0: apys['latest'] = (last_day_data['feesUSD'] / last_day_data[
                'tvlUSD']) * 365 * 100
            for days in [7, 30]:
                window_data = df_day.loc[df_day.index >= (end_date - timedelta(days=days - 1))].tail(days)
                if not window_data.empty and len(window_data) >= min(days, 3):
                    total_fees = window_data['feesUSD'].sum();
                    avg_tvl = window_data['tvlUSD'].mean()
                    if avg_tvl > 0:
                        days_to_annualize = max(1, (window_data.index.max() - window_data.index.min()).days + 1,
                                                len(window_data))
                        if days_to_annualize > 0: apys[f'avg_{days}d'] = (total_fees / avg_tvl) * (
                                365 / days_to_annualize) * 100
            df_day_vol = df_day[df_day['token1Price'] > 1e-18].copy()
            if len(df_day_vol) >= 2:
                df_day_vol['log_return'] = np.log(df_day_vol['token1Price'] / df_day_vol['token1Price'].shift(1))
                df_vol_clean = df_day_vol.dropna(subset=['log_return'])
                if len(df_vol_clean) >= 2:
                    end_date_vol = df_vol_clean.index.max()
                    for days in [7, 30]:
                        valid_returns = df_vol_clean.loc[
                            df_vol_clean.index >= (end_date_vol - timedelta(days=days - 1)), 'log_return'].tail(
                            days).dropna()
                        if len(valid_returns) >= 2:
                            daily_std_dev = valid_returns.std()
                            if pd.notna(daily_std_dev) and daily_std_dev > 1e-9:
                                volatilities[f'vol_{days}d'] = daily_std_dev * np.sqrt(365) * 100
                            else:
                                volatilities[f'vol_{days}d'] = 0.0
        else:
            processing_warning = "Nenhum dado histórico válido encontrado após limpeza."
    except Exception as e:
        processing_warning = f"Erro ao processar dados históricos: {e}"

    base_return.update({
        "volume_24h_approx_usd": volume_24h, "apys": apys,
        "volatility_7d": volatilities['vol_7d'], "volatility_30d": volatilities['vol_30d'],
        "last_update_subgraph": last_update_timestamp, "warning": processing_warning
    })
    return base_return


@st.cache_data(ttl=600)
def get_all_active_ticks_data(pool_id, token0_decimals, token1_decimals, subgraph_url):
    logger.info(f"Fetching all active ticks (liquidityNet) for pool {pool_id}...")
    all_ticks_data = []
    skip = 0
    max_pages = 20  # Limite para evitar loops infinitos ou queries excessivas
    pages_fetched = 0
    while pages_fetched < max_pages:
        variables = {"poolAddress": pool_id.lower(), "skip": skip}
        response_data = send_graphql_query(subgraph_url, ACTIVE_TICKS_QUERY, variables)
        if not response_data or 'data' not in response_data or not response_data['data'].get('ticks'):
            if pages_fetched == 0 and (not response_data or 'data' not in response_data):
                logger.error(
                    f"Failed to fetch any ticks for pool {pool_id} on first attempt. Response: {response_data}")
                return None  # Retorna None se a primeira tentativa falhar completamente
            logger.warning(
                f"No more ticks found for pool {pool_id} at skip {skip}. Fetched {len(all_ticks_data)} ticks.")
            break  # Sai do loop se não houver mais ticks ou erro após a primeira página
        ticks_page = response_data['data']['ticks']
        if not ticks_page: break  # Segurança adicional
        all_ticks_data.extend(ticks_page)
        if len(ticks_page) < 1000: break  # Última página
        skip += 1000
        pages_fetched += 1
        if pages_fetched >= max_pages:
            logger.warning(
                f"Reached max_pages ({max_pages}) for ticks for pool {pool_id}. Processed {len(all_ticks_data)}.")
            st.warning(
                f"Aviso: Análise de liquidez para {pool_id[:8]}... pode ser incompleta (limitado a {max_pages * 1000} ticks).")
            break
    if not all_ticks_data: return pd.DataFrame(
        columns=['tickIdx', 'liquidityNet', 'price_at_tick', 'active_liquidity_at_tick'])
    processed_data = []
    for tick_info in all_ticks_data:
        try:
            tick_idx = int(tick_info['tickIdx']);
            liquidity_net = int(tick_info['liquidityNet'])
            price = (1.0001 ** tick_idx) * (10 ** (token0_decimals - token1_decimals))
            processed_data.append({'tickIdx': tick_idx, 'liquidityNet': liquidity_net, 'price_at_tick': price})
        except (ValueError, TypeError, OverflowError) as e:
            logger.warning(f"Skipping tick for pool {pool_id}, tickIdx {tick_info.get('tickIdx')}: {e}")
    if not processed_data: return pd.DataFrame(
        columns=['tickIdx', 'liquidityNet', 'price_at_tick', 'active_liquidity_at_tick'])
    df_ticks = pd.DataFrame(processed_data).sort_values(by='tickIdx').reset_index(drop=True)
    df_ticks['active_liquidity_at_tick'] = df_ticks['liquidityNet'].cumsum()
    df_ticks.attrs = {'pool_id': pool_id, 'token0_decimals': token0_decimals, 'token1_decimals': token1_decimals}
    logger.info(f"Successfully processed {len(df_ticks)} active ticks for pool {pool_id}.")
    return df_ticks


def get_liquidity_dataframe(pool_id, width_percent=5):
    logger.info(f"Iniciando busca de distribuição de liquidez (liquidityGross) para pool_id: {pool_id}")
    try:
        transport = RequestsHTTPTransport(url=SUBGRAPH_URL, headers={"Content-Type": "application/json",
                                                                     'Authorization': f'Bearer {GRAPH_API_KEY}'},
                                          retries=3, timeout=60)
        client = Client(transport=transport, fetch_schema_from_transport=False)
        query_gross = gql("""
        query GetPoolLiquidityGrossData($pool_id: ID!) {
          pool(id: $pool_id) {
            token0 { symbol decimals id }
            token1 { symbol decimals id }
            tick
            ticks(first: 1000, orderBy: tickIdx, orderDirection: asc, where: {liquidityGross_gt: "0"}) {
              tickIdx
              liquidityGross
            }
          }
        }
        """)
        result = client.execute(query_gross, variable_values={"pool_id": pool_id.lower()})
        if not result or "pool" not in result or result["pool"] is None: return pd.DataFrame()
        pool_data = result["pool"]
        token0_data = pool_data.get("token0");
        token1_data = pool_data.get("token1");
        current_tick_str = pool_data.get("tick")
        if not all(
                [token0_data, token1_data, token0_data.get("symbol"), token1_data.get("symbol")]): return pd.DataFrame()
        try:
            decimals0 = int(token0_data.get("decimals", 18));
            decimals1 = int(token1_data.get("decimals", 18))
            current_tick = int(current_tick_str) if current_tick_str is not None else None
            current_price = (1.0001 ** current_tick) * (
                    10 ** (decimals0 - decimals1)) if current_tick is not None else None
        except (ValueError, OverflowError) as e:
            logger.error(f"Erro decimais/tick pool {pool_id}: {e}");
            return pd.DataFrame()
        raw_ticks_data = pool_data.get("ticks")
        if not isinstance(raw_ticks_data, list) or not raw_ticks_data: return pd.DataFrame()
        processed_tick_data = []
        for tick_entry in raw_ticks_data:
            if tick_entry is None: continue
            try:
                tick_idx = int(tick_entry.get("tickIdx"));
                liquidity_gross = int(tick_entry.get("liquidityGross"))
                if liquidity_gross <= 0: continue
                price_at_tick = (1.0001 ** tick_idx) * (10 ** (decimals0 - decimals1))
                distance_percent = abs((
                                               price_at_tick - current_price) / current_price * 100) if current_price and current_price > 1e-18 else None
                is_near = distance_percent <= width_percent if distance_percent is not None else False
                liquidity_value_approx = liquidity_gross / (10 ** decimals1)  # Approx value in token1 terms
                processed_tick_data.append({'tick': tick_idx, 'price': price_at_tick, 'liquidity_raw': liquidity_gross,
                                            'liquidity_value_approx': liquidity_value_approx,
                                            'distance_percent': distance_percent, 'is_near': is_near})
            except (ValueError, TypeError, OverflowError) as ve:
                logger.warning(f"Erro processar tick (gross) pool {pool_id}: {ve}")
        if not processed_tick_data: return pd.DataFrame()
        df = pd.DataFrame(processed_tick_data).drop_duplicates(subset=['price']).sort_values(by='tick').reset_index(
            drop=True)
        df.attrs = {"pool_id": pool_id,
                    "token_pair": f"{token0_data.get('symbol', 'T0')}/{token1_data.get('symbol', 'T1')}",
                    "current_price": current_price, "current_tick": current_tick,
                    "decimals": {token0_data.get('symbol', 'T0'): decimals0,
                                 token1_data.get('symbol', 'T1'): decimals1}}
        return df
    except Exception as e:
        logger.error(
            f"Erro get_liquidity_dataframe pool {pool_id}: {e}\n{traceback.format_exc()}");
        return pd.DataFrame()


def calculate_L_per_dollar(P_current, P_min_param, P_max_param):
    # Ensure P_min is always less than P_max for calculations
    P_min = min(P_min_param, P_max_param)
    P_max = max(P_min_param, P_max_param)
    if P_min == P_max and not math.isinf(P_max):  # Handle case where P_min and P_max are identical (not full range)
        P_max = P_min * 1.000001  # Create a tiny range to avoid division by zero if P_current is exactly P_min

    if P_current <= 0 or P_min < 0 or (not math.isinf(P_max) and P_max <= P_min):
        logger.warning(
            f"L_per_dollar: Input inválido P_current={P_current}, P_min_orig={P_min_param}, P_max_orig={P_max_param} -> P_min_calc={P_min}, P_max_calc={P_max}")
        return 0

    P_min_safe = max(P_min, 1e-18)
    if P_current < P_min_safe or (not math.isinf(P_max) and P_current > P_max):
        logger.info(
            f"L_per_dollar: P_current ({P_current:.4f}) FORA da faixa [{P_min_safe:.4f}, {P_max if not math.isinf(P_max) else 'inf'}] -> L=0")
        return 0

    if not math.isinf(P_max) and P_min_safe >= P_max:
        logger.warning(f"L_per_dollar: P_min_safe ({P_min_safe}) >= P_max ({P_max}). Faixa inválida após ajuste.")
        return 0

    try:
        sp = math.sqrt(P_current)
        sa = math.sqrt(P_min_safe)
        sb = math.sqrt(P_max) if not math.isinf(P_max) else float('inf')

        if P_min_safe == P_max:  # Should be caught by P_max <= P_min earlier, but as a safeguard for single point after adjustments
            logger.warning(
                f"L_per_dollar: Faixa de um único ponto P_min=P_max={P_min_safe}. Retornando L grande (arbitrário).")
            return 1e18

        denominator_term1 = 2 * sp
        denominator_term2 = sa
        denominator_term3 = (P_current / sb) if not math.isinf(sb) else 0

        denominator = denominator_term1 - denominator_term2 - denominator_term3

        logger.debug(
            f"L_per_dollar calc: P_curr={P_current:.4f}, P_min_calc={P_min_safe:.4f}, P_max_calc={P_max if not math.isinf(P_max) else 'inf'}")
        logger.debug(f"  sqrt_terms: sp={sp:.4f}, sa={sa:.4f}, sb={sb if not math.isinf(sb) else 'inf'}")
        logger.debug(
            f"  denom_terms: 2*sp={denominator_term1:.4f}, sa={denominator_term2:.4f}, P/sb={denominator_term3:.4f}")
        logger.debug(
            f"  denominator = {denominator_term1:.4f} - {denominator_term2:.4f} - {denominator_term3:.4f} = {denominator:.4f}")

        if denominator <= 1e-18:
            logger.warning(
                f"L_per_dollar: Denominador ({denominator:.2E}) muito pequeno/negativo. Retornando L grande.")
            return 1e18

        result = 1.0 / denominator
        logger.info(
            f"L_per_dollar: P_curr={P_current:.4f}, Range=[{P_min_safe:.4f}-{P_max if not math.isinf(P_max) else 'inf'}], L_per_dollar={result:.4E}")
        return result if isinstance(result, (int, float)) and math.isfinite(result) and result >= 0 else 0
    except (ValueError, OverflowError) as e:
        logger.error(f"Erro L_per_dollar (P={P_current}, min_orig={P_min_param}, max_orig={P_max_param}): {e}");
        return 0


def estimate_time_in_range(P_current, P_min_param, P_max_param, volatility_annualized_percent):
    P_min = min(P_min_param, P_max_param)
    P_max = max(P_min_param, P_max_param)
    if P_min == P_max and not math.isinf(P_max): P_max = P_min * 1.000001

    if volatility_annualized_percent is None or volatility_annualized_percent <= 1e-6: return 100.0 if P_min < P_current < P_max else 0.0
    if P_current <= 0 or P_min < 0 or (not math.isinf(P_max) and P_max <= P_min): return None
    try:
        vol_daily = (volatility_annualized_percent / 100.0) / math.sqrt(365)
        if vol_daily <= 0: return 100.0 if P_min < P_current < P_max else 0.0
        P_min_safe = max(P_min, 1e-18)
        dist_upper_log = math.log(P_max / P_current) if not math.isinf(P_max) and P_max / P_current > 0 else float(
            'inf')
        dist_lower_log = math.log(P_current / P_min_safe) if P_current / P_min_safe > 0 else float('-inf')
        if dist_lower_log == float('-inf'): return None
        z_upper = dist_upper_log / vol_daily;
        z_lower = dist_lower_log / vol_daily
        return max(0.0, min(1.0, norm.cdf(z_upper) - norm.cdf(-z_lower))) * 100.0
    except Exception as e:
        logger.error(f"Erro estimate_time_in_range: {e}");
        return None


MIN_DATA_DRIVEN_EFFICIENCY_THRESHOLD = 0.01  # Percent. If calculated DDE is below this, fallback to manual.


def simulate_v3_fees_subgraph_data(
        investment_usd, user_min_price_p0_p1_param, user_max_price_p0_p1_param, current_market_price_p0_p1,
        selected_pool_fee_apr_percent, pool_tvl_usd, volatility_annualized_percent,
        capture_efficiency_percent,  # This is the manual slider input
        manual_tir_percent=None,
        active_liquidity_df=None, use_data_driven_capture_efficiency=False, current_tick_for_pool=None
):
    # Ensure min_price is always less than max_price for the simulation logic
    user_min_price_p0_p1 = min(user_min_price_p0_p1_param, user_max_price_p0_p1_param)
    user_max_price_p0_p1 = max(user_min_price_p0_p1_param, user_max_price_p0_p1_param)
    if user_min_price_p0_p1 == user_max_price_p0_p1 and not math.isinf(user_max_price_p0_p1):
        user_max_price_p0_p1 = user_min_price_p0_p1 * 1.000001  # Ensure a tiny valid range

    logger.info(
        f"[SIM_INIT] Invest=${investment_usd}, Range_Orig=[{user_min_price_p0_p1_param:.4f}-{user_max_price_p0_p1_param:.4f}], Range_Calc=[{user_min_price_p0_p1:.4f}-{user_max_price_p0_p1:.4f}], P_market={current_market_price_p0_p1:.4f}, APR_base={selected_pool_fee_apr_percent:.2f}%, TVL=${pool_tvl_usd:,.0f}, UseDataDrivenEff={use_data_driven_capture_efficiency}")

    results_base = {
        "status": "OK", "warning_message": None, "v3_calculation_notes": [],
        "investment_usd": investment_usd, "range_p0_p1_min": user_min_price_p0_p1,
        "range_p0_p1_max": user_max_price_p0_p1,
        "market_price_p0_p1": current_market_price_p0_p1, "selected_apr_input": selected_pool_fee_apr_percent,
        "tvl_pool": pool_tvl_usd,
        "tir_used": 0.0, "tir_source": "N/A",
        "capture_efficiency_input_manual": capture_efficiency_percent,
        "final_capture_efficiency_used": capture_efficiency_percent,
        "data_driven_efficiency_calculated_value": None,
        "data_driven_efficiency_source": "Manual (Slider)",
        "calculated_tir": None, "volatility_used": None,
        "total_daily_fees_pool_apr": 0.0,
        "fees_daily_tvl_share": 0.0, "fees_weekly_tvl_share": 0.0,
        "fees_monthly_tvl_share": 0.0, "fees_annual_tvl_share": 0.0, "apr_tvl_share": 0.0,
        "amplification_factor": 0.0, "raw_potential_apr": 0.0,
        "fees_daily_v3_adjusted": 0.0, "fees_weekly_v3_adjusted": 0.0,
        "fees_monthly_v3_adjusted": 0.0, "fees_annual_v3_adjusted": 0.0, "apr_v3_adjusted": 0.0,
    }

    if not (user_min_price_p0_p1 < user_max_price_p0_p1 if not math.isinf(
            user_max_price_p0_p1) else True) or user_min_price_p0_p1 < 0 or investment_usd <= 0 or current_market_price_p0_p1 <= 0 or pool_tvl_usd <= 0 or not (
            0 <= capture_efficiency_percent <= 100) or selected_pool_fee_apr_percent < 0:
        results_base["status"] = "Erro"
        results_base["warning_message"] = "Input inválido detectado (ex: investimento, preços, APR, TVL)."
        logger.error(f"[SIM_ERROR] Input inválido: {results_base['warning_message']}")
        return results_base

    if selected_pool_fee_apr_percent == 0:
        results_base["v3_calculation_notes"].append(
            "APR base da pool é 0%, portanto, taxas V3 estimadas também serão 0%.")
        logger.info("[SIM_V3_NOTE] APR base da pool é 0%. Taxas V3 serão 0.")

    tir_to_use = None;
    calculated_tir_percent = None
    if volatility_annualized_percent is not None and volatility_annualized_percent >= 0:
        calculated_tir_percent = estimate_time_in_range(current_market_price_p0_p1, user_min_price_p0_p1,
                                                        user_max_price_p0_p1, volatility_annualized_percent)
        if calculated_tir_percent is not None:
            tir_to_use = calculated_tir_percent
            results_base["tir_source"] = "Calculado (Volatilidade 7d)"
            results_base["volatility_used"] = volatility_annualized_percent
    if tir_to_use is None and manual_tir_percent is not None:
        tir_to_use = float(manual_tir_percent)
        results_base["tir_source"] = "Manual (Fallback)"

    if tir_to_use is None:
        results_base["status"] = "Erro"
        results_base["warning_message"] = "Não foi possível determinar TiR."
        logger.error(f"[SIM_ERROR] Falha ao determinar TiR.")
        return results_base
    results_base["tir_used"] = tir_to_use
    results_base["calculated_tir"] = calculated_tir_percent
    logger.info(f"[SIM_TIR] TiR usado: {tir_to_use:.2f}% (Fonte: {results_base['tir_source']})")

    avg_daily_return_rate = (selected_pool_fee_apr_percent / 100.0) / 365.0
    estimated_total_daily_fees_pool = pool_tvl_usd * avg_daily_return_rate
    tvl_share = investment_usd / pool_tvl_usd if pool_tvl_usd > 0 else 0
    fees_daily_tvl_share = estimated_total_daily_fees_pool * tvl_share
    apr_tvl_share = selected_pool_fee_apr_percent

    results_base["total_daily_fees_pool_apr"] = estimated_total_daily_fees_pool
    results_base["fees_daily_tvl_share"] = fees_daily_tvl_share
    results_base["fees_weekly_tvl_share"] = fees_daily_tvl_share * 7
    results_base["fees_monthly_tvl_share"] = fees_daily_tvl_share * 30.4375
    results_base["fees_annual_tvl_share"] = fees_daily_tvl_share * 365
    results_base["apr_tvl_share"] = apr_tvl_share
    logger.info(f"[SIM_V2_RES] APR V2 (TVL Share): {apr_tvl_share:.2f}%, Taxas Diárias V2: ${fees_daily_tvl_share:.4f}")

    is_in_range_at_current = (user_min_price_p0_p1 < current_market_price_p0_p1 < user_max_price_p0_p1)
    results_base["status"] = "Dentro da Faixa" if is_in_range_at_current else "Fora da Faixa"

    if not is_in_range_at_current:
        results_base["warning_message"] = "Preço de mercado atual está fora da faixa definida."
        results_base["v3_calculation_notes"].append("Posição fora da faixa de preço atual. Taxas V3 são 0.")
        logger.info(f"[SIM_V3_NOTE] Fora da Faixa. Taxas V3 e APR V3 serão 0. {results_base['warning_message']}")
        return results_base

    logger.info(f"[SIM_V3_CALC] Posição DENTRO DA FAIXA. Prosseguindo com cálculos V3.")

    if use_data_driven_capture_efficiency and active_liquidity_df is not None and not active_liquidity_df.empty and current_tick_for_pool is not None:
        logger.info("[DD_EFF] Tentando calcular eficiência de captura baseada em dados...")
        data_driven_success = False
        try:
            L_user_per_dollar_dd = calculate_L_per_dollar(current_market_price_p0_p1, user_min_price_p0_p1,
                                                          user_max_price_p0_p1)
            logger.info(
                f"[DD_EFF] P_current={current_market_price_p0_p1:.4f}, P_min_calc={user_min_price_p0_p1:.4f}, P_max_calc={user_max_price_p0_p1:.4f}")
            logger.info(
                f"[DD_EFF] L_user_per_dollar_dd (para eficiência): {L_user_per_dollar_dd:.4E}, Investment: ${investment_usd:,.2f}")

            if L_user_per_dollar_dd > 1e-18 and investment_usd > 0:
                L_user_units = L_user_per_dollar_dd * investment_usd
                logger.info(f"[DD_EFF] L_user_units: {L_user_units:.4E}")

                pool_ticks_at_or_before_current = active_liquidity_df[
                    active_liquidity_df['tickIdx'] <= current_tick_for_pool]
                logger.info(
                    f"[DD_EFF] Encontrado {len(pool_ticks_at_or_before_current)} ticks da pool no ou antes do tick atual da pool {current_tick_for_pool}")

                if not pool_ticks_at_or_before_current.empty:
                    L_pool_existing_at_current_tick = pool_ticks_at_or_before_current.iloc[-1][
                        'active_liquidity_at_tick']
                    logger.info(
                        f"[DD_EFF] L_pool_existing_at_current_tick (L cumulativa da pool até o tick {current_tick_for_pool}): {L_pool_existing_at_current_tick:.4E}")

                    if L_user_units > 1e-18:
                        if L_pool_existing_at_current_tick >= 0:
                            total_L_at_tick_with_user = L_pool_existing_at_current_tick + L_user_units
                            logger.info(
                                f"[DD_EFF] Total L (L_pool_existente + L_user) no tick atual: {total_L_at_tick_with_user:.4E}")

                            if total_L_at_tick_with_user > 1e-18:
                                data_driven_eff_raw = L_user_units / total_L_at_tick_with_user
                                logger.info(
                                    f"[DD_EFF] data_driven_eff_raw calculada (L_user / Total_L_com_user): {data_driven_eff_raw:.6f}")

                                data_driven_efficiency_value_calc = min(max(data_driven_eff_raw * 100.0, 0.0),
                                                                        100.0)
                                results_base[
                                    "data_driven_efficiency_calculated_value"] = data_driven_efficiency_value_calc

                                if data_driven_efficiency_value_calc < MIN_DATA_DRIVEN_EFFICIENCY_THRESHOLD:
                                    logger.warning(
                                        f"[DD_EFF_FALLBACK] Eficiência calculada ({data_driven_efficiency_value_calc:.4f}%) é menor que o limite ({MIN_DATA_DRIVEN_EFFICIENCY_THRESHOLD}%) e será desconsiderada. Usando valor manual.")
                                    results_base[
                                        "data_driven_efficiency_source"] = f"Manual (Dados < {MIN_DATA_DRIVEN_EFFICIENCY_THRESHOLD}%)"
                                    results_base["v3_calculation_notes"].append(
                                        f"Eficiência por dados ({data_driven_efficiency_value_calc:.4f}%) muito baixa, usando manual ({capture_efficiency_percent:.2f}%).")
                                else:
                                    results_base["final_capture_efficiency_used"] = data_driven_efficiency_value_calc
                                    results_base["data_driven_efficiency_source"] = (
                                        f"Dados ({data_driven_efficiency_value_calc:.2f}%)"
                                    )
                                    logger.info(
                                        f"[DD_EFF] Usando eficiência baseada em dados: {results_base['final_capture_efficiency_used']:.2f}%")
                                    data_driven_success = True  # Mark success only if DDE is used
                            else:
                                logger.warning(
                                    "[DD_EFF_FALLBACK] Total_L_at_tick_with_user é zero ou muito pequeno. Usando eficiência manual (slider). ")
                                results_base["data_driven_efficiency_source"] = "Manual (L_total c/ user zero)"
                                results_base["v3_calculation_notes"].append(
                                    "Eficiência por dados não pôde ser usada (L_total zero). Usando manual.")
                        else:
                            logger.warning(
                                f"[DD_EFF_FALLBACK] L_pool_existing_at_current_tick é negativo ({L_pool_existing_at_current_tick:.4E}). Erro nos dados. Usando eficiência manual (slider).")
                            results_base["data_driven_efficiency_source"] = "Manual (L_pool_exist < 0)"
                            results_base["v3_calculation_notes"].append(
                                "Eficiência por dados não pôde ser usada (L_pool < 0). Usando manual.")
                    else:
                        logger.warning(
                            "[DD_EFF_FALLBACK] L_user_units (para eficiência) é zero ou muito pequeno. Usando eficiência manual (slider).")
                        results_base["data_driven_efficiency_source"] = "Manual (L_user_dd zero)"
                        results_base["v3_calculation_notes"].append(
                            "Eficiência por dados não pôde ser usada (L_user zero). Usando manual.")
                else:
                    logger.warning(
                        f"[DD_EFF_FALLBACK] Nenhum tick da pool encontrado no ou antes do tick atual da pool ({current_tick_for_pool}). Usando eficiência manual (slider).")
                    results_base["data_driven_efficiency_source"] = "Manual (Pool ticks não encontrados)"
                    results_base["v3_calculation_notes"].append(
                        "Eficiência por dados não pôde ser usada (pool ticks não encontrados). Usando manual.")
            else:
                logger.warning(
                    f"[DD_EFF_FALLBACK] L_user_per_dollar_dd ({L_user_per_dollar_dd:.4E}) ou investment_usd ({investment_usd:,.2f}) é zero/inválido para cálculo de eficiência por dados. Usando manual.")
                results_base["data_driven_efficiency_source"] = "Manual (L_user_per_dollar_dd zero)"
                results_base["v3_calculation_notes"].append(
                    "Eficiência por dados não pôde ser usada (L_user_per_dollar zero). Usando manual.")
        except Exception as e:
            logger.error(
                f"[DD_EFF_FALLBACK] Exceção ao calcular eficiência baseada em dados: {e}\n{traceback.format_exc()}. Usando eficiência manual (slider).")
            results_base["data_driven_efficiency_source"] = f"Manual (Exceção DD: {str(e)[:30]}...)"
            results_base["v3_calculation_notes"].append(
                f"Erro no cálculo de eficiência por dados: {str(e)[:50]}. Usando manual.")

        if not data_driven_success:
            logger.info(
                "[DD_EFF_FALLBACK_FINAL] Falha ao usar eficiência baseada em dados, revertendo para valor manual do slider.")
            results_base[
                "final_capture_efficiency_used"] = capture_efficiency_percent  # Ensure fallback to slider value
            # Source is already set to manual or a specific failure reason

    L_per_dollar_for_amplification = calculate_L_per_dollar(current_market_price_p0_p1, user_min_price_p0_p1,
                                                            user_max_price_p0_p1)
    L_per_dollar_reference_v2 = 0.5 / math.sqrt(
        current_market_price_p0_p1) if current_market_price_p0_p1 > 1e-18 else 0

    logger.info(
        f"[SIM_V3_AMP] L_per_dollar_for_amplification: {L_per_dollar_for_amplification:.4E}, L_per_dollar_reference_v2: {L_per_dollar_reference_v2:.4E}")

    amplification_factor_calc = 0.0
    if L_per_dollar_for_amplification > 1e-12 and L_per_dollar_reference_v2 > 1e-12:
        amplification_factor_calc = min(L_per_dollar_for_amplification / L_per_dollar_reference_v2, 10000.0)
    else:
        note = f"Fator de amplificação é 0 porque L_per_dollar_user ({L_per_dollar_for_amplification:.2E}) ou L_per_dollar_v2_ref ({L_per_dollar_reference_v2:.2E}) é muito baixo."
        if not any(note_item == note for note_item in results_base["v3_calculation_notes"]):
            results_base["v3_calculation_notes"].append(note)
        logger.warning(f"[SIM_V3_AMP_WARN] {note}")

    results_base["amplification_factor"] = amplification_factor_calc
    logger.info(f"[SIM_V3_AMP_RES] amplification_factor_calc: {amplification_factor_calc:.4f}")

    user_share_factor_amplified = amplification_factor_calc * tvl_share
    user_potential_daily_fees_raw = estimated_total_daily_fees_pool * user_share_factor_amplified
    raw_potential_apr_percent_calc = (
                                                 user_potential_daily_fees_raw * 365 / investment_usd) * 100 if investment_usd > 0 else 0
    results_base["raw_potential_apr"] = raw_potential_apr_percent_calc
    logger.info(
        f"[SIM_V3_RAWFEES] User Share Amplified: {user_share_factor_amplified:.4E}, Potential Daily Fees Raw: ${user_potential_daily_fees_raw:.4E}, Raw Potential APR: {raw_potential_apr_percent_calc:.2f}%")

    if selected_pool_fee_apr_percent > 0 and estimated_total_daily_fees_pool == 0:
        note = "Taxas diárias totais da pool estimadas são 0 (verificar TVL ou APR base)."
        if not any(note_item == note for note_item in results_base["v3_calculation_notes"]):
            results_base["v3_calculation_notes"].append(note)
            logger.warning(f"[SIM_V3_WARN] {note}")

    if selected_pool_fee_apr_percent > 0 and amplification_factor_calc == 0 and user_potential_daily_fees_raw == 0:
        if not any("Fator de amplificação é 0" in note for note in results_base["v3_calculation_notes"]):
            note = "Taxas V3 brutas são 0 devido ao fator de amplificação ser 0."
            results_base["v3_calculation_notes"].append(note)
            logger.warning(f"[SIM_V3_WARN] {note}")

    tir_factor = tir_to_use / 100.0
    efficiency_factor_final = results_base["final_capture_efficiency_used"] / 100.0

    logger.info(
        f"[SIM_V3_ADJ_FACTORS] TiR Factor: {tir_factor:.4f}, Final Efficiency Factor: {efficiency_factor_final:.4f}")

    fees_daily_v3_adjusted_calc = user_potential_daily_fees_raw * tir_factor * efficiency_factor_final
    apr_v3_adjusted_calc = (fees_daily_v3_adjusted_calc * 365 / investment_usd) * 100 if investment_usd > 0 else 0

    results_base["fees_daily_v3_adjusted"] = fees_daily_v3_adjusted_calc
    results_base["fees_weekly_v3_adjusted"] = fees_daily_v3_adjusted_calc * 7
    results_base["fees_monthly_v3_adjusted"] = fees_daily_v3_adjusted_calc * 30.4375
    results_base["fees_annual_v3_adjusted"] = fees_daily_v3_adjusted_calc * 365
    results_base["apr_v3_adjusted"] = apr_v3_adjusted_calc

    if results_base["apr_v3_adjusted"] == 0 and selected_pool_fee_apr_percent > 0 and is_in_range_at_current and not \
    results_base["v3_calculation_notes"]:
        results_base["v3_calculation_notes"].append(
            f"APR V3 Ajustado é 0. Detalhes: TiR={tir_factor * 100:.1f}%, Eficiência={efficiency_factor_final * 100:.1f}%, Amplificação={amplification_factor_calc:.2f}x, Taxas Brutas=${user_potential_daily_fees_raw:.2E}."
        )

    logger.info(
        f"[SIM_FINAL_V3_RES] APR V3 Ajustado: {results_base['apr_v3_adjusted']:.2f}%, Taxas Diárias V3 Ajustadas: ${results_base['fees_daily_v3_adjusted']:.4f}")
    logger.info(
        f"  Detalhes V3: Amplificação={results_base['amplification_factor']:.2f}x, Eficiência Final Usada={results_base['final_capture_efficiency_used']:.2f}% ({results_base['data_driven_efficiency_source']}), TiR Usado={results_base['tir_used']:.2f}% ({results_base['tir_source']})")
    if results_base["v3_calculation_notes"]: logger.info(
        f"  Notas Cálculo V3: {' | '.join(results_base['v3_calculation_notes'])}")
    return results_base


# --- Streamlit UI ---
st.set_page_config(page_title="Simulador LP V3", layout="wide")
tab1, tab2, tab3 = st.tabs(
    ["📊 Simulador Comparativo", "💧 Distribuição de Liquidez (Ticks)", "ℹ️ Explicação dos Métodos"])

# Initialize session state
for key, default_val in [
    ('all_pools_data', {}), ('selected_pool_key', None), ('detailed_pool_data', None),
    ('liquidity_distribution_data', None), ('active_liquidity_df', None),
    ('active_liquidity_df_pool_id', None), ('investment_usd', 1000.0),
    ('selected_apr_source', "Média 7d"), ('price_perspective', None),
    ('min_price_input', 0.0), ('max_price_input', 0.0), ('market_price_input', 0.0),
    ('capture_efficiency_input', 15.0), ('manual_tir_estimate', 50),
    ('show_manual_tir', False), ('simulation_result', None),
    ('data_loaded_successfully', False), ('use_data_driven_efficiency', False)
]:
    if key not in st.session_state: st.session_state[key] = default_val


def format_price(value):
    if value is None: return "N/A"
    if math.isinf(value): return "∞"
    if 0 < abs(value) < 1e-8: return f"{value:.2e}"
    if abs(value) >= 1e9: return f"{value:.2e}"
    if abs(value) < 1: return f"{value:.6f}"
    return f"{value:.2f}"


def format_percentage(value): return f"{value:.2f}%" if value is not None else "N/A"


with tab1:
    st.title("Simulador Comparativo de LP Uniswap V3")
    with st.sidebar:
        st.header("1. Seleção da Pool")
        if not st.session_state.all_pools_data:
            with st.spinner("Carregando lista de pools..."):
                st.session_state.all_pools_data = get_top_pools_subgraph(SUBGRAPH_URL)
        if not st.session_state.all_pools_data: st.error("Falha ao carregar pools."); st.stop()

        pool_display_names = {k: v["display_name"] for k, v in st.session_state.all_pools_data.items()}
        selected_pool_id_from_ui = st.selectbox("Escolha uma Pool:",
                                                options=list(st.session_state.all_pools_data.keys()),
                                                format_func=lambda pool_id: pool_display_names.get(pool_id, pool_id),
                                                key="sb_selected_pool_id",
                                                index=0 if st.session_state.all_pools_data else None)

        if selected_pool_id_from_ui and (
                st.session_state.selected_pool_key != selected_pool_id_from_ui or not st.session_state.data_loaded_successfully):
            st.session_state.selected_pool_key = selected_pool_id_from_ui
            st.session_state.data_loaded_successfully = False
            st.session_state.simulation_result = None
            st.session_state.liquidity_distribution_data = None
            st.session_state.active_liquidity_df = None
            st.session_state.active_liquidity_df_pool_id = None
            with st.spinner(
                    f"Carregando dados para {st.session_state.all_pools_data[selected_pool_id_from_ui]['display_name']}..."):
                st.session_state.detailed_pool_data = get_detailed_pool_data(selected_pool_id_from_ui, SUBGRAPH_URL)
            if st.session_state.detailed_pool_data:
                st.session_state.data_loaded_successfully = True;
                dpd = st.session_state.detailed_pool_data
                st.session_state.price_perspective = f"{dpd['token0_symbol']}/{dpd['token1_symbol']}"
                st.session_state.market_price_input = dpd.get("current_price_p0_p1") or 0.0
                if st.session_state.market_price_input > 0:
                    st.session_state.min_price_input = st.session_state.market_price_input * 0.90
                    st.session_state.max_price_input = st.session_state.market_price_input * 1.10
                st.success("Dados da pool carregados!")
                if dpd.get("warning"): st.warning(f"Aviso dados pool: {dpd['warning']}")
                st.rerun()
            else:
                st.error("Falha ao carregar dados detalhados da pool. Verifique a console para logs de erro da API.")

    if not st.session_state.data_loaded_successfully or not st.session_state.detailed_pool_data: st.info(
        "Selecione uma pool na barra lateral para começar."); st.stop()
    dpd = st.session_state.detailed_pool_data

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.subheader(f"Pool: {dpd['token0_symbol']}/{dpd['token1_symbol']} ({dpd['feeTier'] * 100:.2f}%)");
        st.metric(
            "TVL (USD)", f"${dpd.get('current_tvl_usd', 0):,.0f}");
        st.metric(
            f"Preço ({st.session_state.price_perspective})", format_price(dpd.get('current_price_p0_p1')))
    with col_info2:
        st.metric("Volume 24h (USD)", f"${dpd.get('volume_24h_approx_usd', 0):,.0f}");
        st.metric("APR Base (7d)",
                  format_percentage(
                      dpd['apys'].get(
                          'avg_7d')));
        st.caption(
            f"Atualizado: {dpd.get('last_update_subgraph', 'N/A')}")
    if dpd.get("warning"): st.warning(f"Aviso sobre dados da pool: {dpd['warning']}")
    st.divider()

    st.header("2. Parâmetros da Simulação")
    col_param1, col_param2, col_param3 = st.columns(3)
    with col_param1:
        st.session_state.investment_usd = st.number_input("Investimento (USD):", min_value=1.0,
                                                          value=st.session_state.investment_usd, step=100.0,
                                                          key="num_investment_usd")
        apr_options = {"Último Dia": dpd['apys'].get('latest', 0.0), "Média 7d": dpd['apys'].get('avg_7d', 0.0),
                       "Média 30d": dpd['apys'].get('avg_30d', 0.0)}
        st.session_state.selected_apr_source = st.selectbox("APR Base para Cálculo:", options=list(apr_options.keys()),
                                                            index=list(apr_options.keys()).index(
                                                                st.session_state.selected_apr_source) if st.session_state.selected_apr_source in apr_options else 1,
                                                            key="sb_selected_apr_source")
        selected_apr_value_for_calc = apr_options[st.session_state.selected_apr_source]
    with col_param2:
        st.markdown(f"**Faixa de Preço ({st.session_state.price_perspective})**")
        # Ensure min_price_input is always <= max_price_input from UI
        min_val_ui = st.number_input(f"Preço Mínimo:", min_value=0.0,
                                     value=st.session_state.min_price_input, format="%.6f",
                                     key="num_min_price_ui")
        max_val_ui = st.number_input(f"Preço Máximo:", min_value=0.0,
                                     value=st.session_state.max_price_input, format="%.6f",
                                     key="num_max_price_ui")
        st.session_state.min_price_input = min(min_val_ui, max_val_ui)
        st.session_state.max_price_input = max(min_val_ui, max_val_ui)
        if st.session_state.min_price_input == st.session_state.max_price_input and not math.isinf(
                st.session_state.max_price_input):
            st.session_state.max_price_input = st.session_state.min_price_input * 1.000001  # Ensure a tiny valid range if equal

        if st.button("Faixa Completa (V2-like)",
                     key="btn_full_range"):
            st.session_state.min_price_input = 0.0
            st.session_state.max_price_input = float('inf')
            # Update UI keys directly to reflect change if button is pressed
            st.session_state.num_min_price_ui = 0.0
            st.session_state.num_max_price_ui = float('inf')
            st.rerun()

    with col_param3:
        st.session_state.use_data_driven_efficiency = st.checkbox("Usar eficiência de captura baseada em dados (Beta)",
                                                                  value=st.session_state.use_data_driven_efficiency,
                                                                  key="cb_use_data_driven_efficiency",
                                                                  help="Calcula eficiência pela sua participação na liquidez ativa no tick atual. Requer carregamento de todos os ticks ativos, pode ser lento.")
        if st.session_state.use_data_driven_efficiency:
            if dpd and (
                    st.session_state.active_liquidity_df is None or st.session_state.active_liquidity_df_pool_id != dpd[
                'pool_id']):
                with st.spinner(f"Carregando liquidez ativa para {dpd['token0_symbol']}/{dpd['token1_symbol']}..."):
                    st.session_state.active_liquidity_df = get_all_active_ticks_data(dpd['pool_id'],
                                                                                     dpd['token0_decimals'],
                                                                                     dpd['token1_decimals'],
                                                                                     SUBGRAPH_URL)
                    st.session_state.active_liquidity_df_pool_id = dpd['pool_id']
                    if st.session_state.active_liquidity_df is None or st.session_state.active_liquidity_df.empty:
                        st.warning("Falha ao carregar liquidez ativa. Usando slider manual.")
                        st.session_state.use_data_driven_efficiency = False  # Fallback
                    else:
                        st.success("Liquidez ativa carregada.")
                    st.rerun()  # Rerun to update UI state, especially disabled state of slider

        manual_slider_disabled = (st.session_state.use_data_driven_efficiency and
                                  st.session_state.active_liquidity_df is not None and
                                  not st.session_state.active_liquidity_df.empty and
                                  st.session_state.active_liquidity_df_pool_id == dpd['pool_id'])

        st.session_state.capture_efficiency_input = st.slider(
            "Eficiência de Captura Manual (%):",
            min_value=0.0, max_value=100.0,
            value=st.session_state.capture_efficiency_input,
            step=1.0, key="sl_capture_efficiency",
            help="Usado se 'baseada em dados' estiver desmarcado ou falhar o carregamento/cálculo.",
            disabled=manual_slider_disabled
        )
        volatility_7d_for_tir = dpd.get('volatility_7d')
        st.session_state.show_manual_tir = volatility_7d_for_tir is None or volatility_7d_for_tir <= 1e-6
        if st.session_state.show_manual_tir:
            st.session_state.manual_tir_estimate = st.slider("Estimativa Manual de TiR (%):", min_value=0,
                                                             max_value=100, value=st.session_state.manual_tir_estimate,
                                                             step=1, key="sl_manual_tir")
        else:
            st.session_state.manual_tir_estimate = None

    if st.button("🚀 Simular Rendimentos", type="primary", key="btn_simulate"):
        # Final check for min < max before simulation, using the potentially corrected session_state values
        if st.session_state.min_price_input >= st.session_state.max_price_input and not math.isinf(
                st.session_state.max_price_input):
            st.error("Preço Mínimo deve ser menor que Máximo.")
        elif st.session_state.market_price_input <= 0:
            st.error("Preço de mercado inválido para simulação.")
        else:
            with st.spinner("Calculando simulação..."):
                active_liq_df_to_pass = None
                use_dd_eff_to_pass = False
                if (st.session_state.use_data_driven_efficiency and
                        st.session_state.active_liquidity_df is not None and
                        not st.session_state.active_liquidity_df.empty and
                        st.session_state.active_liquidity_df_pool_id == dpd['pool_id']):
                    active_liq_df_to_pass = st.session_state.active_liquidity_df
                    use_dd_eff_to_pass = True

                st.session_state.simulation_result = simulate_v3_fees_subgraph_data(
                    investment_usd=st.session_state.investment_usd,
                    user_min_price_p0_p1_param=st.session_state.min_price_input,  # Pass corrected values
                    user_max_price_p0_p1_param=st.session_state.max_price_input,  # Pass corrected values
                    current_market_price_p0_p1=st.session_state.market_price_input,
                    selected_pool_fee_apr_percent=selected_apr_value_for_calc,
                    pool_tvl_usd=dpd.get('current_tvl_usd', 0.0),
                    volatility_annualized_percent=volatility_7d_for_tir,
                    capture_efficiency_percent=st.session_state.capture_efficiency_input,
                    manual_tir_percent=st.session_state.manual_tir_estimate,
                    active_liquidity_df=active_liq_df_to_pass,
                    use_data_driven_capture_efficiency=use_dd_eff_to_pass,
                    current_tick_for_pool=dpd.get('current_tick')
                )

    if st.session_state.simulation_result:
        res = st.session_state.simulation_result;
        st.divider();
        st.header("📈 Resultados da Simulação")
        if res["status"] == "Erro":
            st.error(f"Erro na Simulação: {res.get('warning_message', 'Detalhe não disponível.')}")
        else:
            if res.get("warning_message") and res["status"] != "Fora da Faixa":
                st.warning(f"Aviso na Simulação: {res['warning_message']}")
            if res.get("v3_calculation_notes"):
                notes_str = "\n- ".join(res["v3_calculation_notes"])
                st.info(f"Notas sobre o cálculo V3:\n- {notes_str}")

            st.subheader(f"Comparativo de Estratégias de LP (Status: {res['status']})")
            res_cols = st.columns(2)
            with res_cols[0]:
                st.markdown("#### V2 (Faixa Completa / TVL Share)");
                st.metric("APR (V2)", format_percentage(
                    res['apr_tvl_share']));
                st.metric("Taxas Diárias (V2)", f"${res['fees_daily_tvl_share']:.4f}")
            with res_cols[1]:
                st.markdown("#### V3 (Liquidez Concentrada Ajustada)");
                st.metric("APR (V3 Ajustado)",
                          format_percentage(res.get(
                              'apr_v3_adjusted', 0.0)));
                st.metric(
                    "Taxas Diárias (V3 Ajustado)", f"${res.get('fees_daily_v3_adjusted', 0.0):.4f}")
            st.subheader("Detalhes da Simulação V3 Ajustada")
            detail_cols = st.columns(3)
            with detail_cols[0]:
                st.metric("Fator de Amplificação", f"{res.get('amplification_factor', 0.0):.2f}x")
            with detail_cols[1]:
                st.metric("Tempo na Faixa (TiR) Usado", format_percentage(res.get('tir_used', 0.0)));
                st.caption(
                    f"Fonte: {res.get('tir_source', 'N/A')}")
            with detail_cols[2]:
                st.metric("Eficiência de Captura Usada",
                          format_percentage(res.get('final_capture_efficiency_used', 0.0)))
                st.caption(f"Fonte: {res.get('data_driven_efficiency_source', 'N/A')}")

with tab2:
    st.header("💧 Distribuição de Liquidez da Pool (Baseado em Ticks Gross)")
    if not st.session_state.data_loaded_successfully or not st.session_state.detailed_pool_data: st.info(
        "Selecione uma pool na aba 'Simulador'."); st.stop()
    current_pool_id_for_liq_gross = st.session_state.detailed_pool_data["pool_id"]
    if st.session_state.liquidity_distribution_data is None or st.session_state.liquidity_distribution_data.attrs.get(
            "pool_id") != current_pool_id_for_liq_gross:
        with st.spinner(
                f"Carregando liquidez (gross) para {st.session_state.detailed_pool_data['token0_symbol']}/{st.session_state.detailed_pool_data['token1_symbol']}..."):
            st.session_state.liquidity_distribution_data = get_liquidity_dataframe(current_pool_id_for_liq_gross)
    df_liq = st.session_state.liquidity_distribution_data
    if df_liq is not None and not df_liq.empty:
        st.markdown(
            f"**Pool:** {df_liq.attrs.get('token_pair', 'N/A')} | **Preço Atual:** {format_price(df_liq.attrs.get('current_price', 'N/A'))} (Tick: {df_liq.attrs.get('current_tick', 'N/A')})")
        proximity_percent = st.slider("Destacar Ticks Próximos (%):", 1, 50, 5, key="sl_liq_proximity")
        current_price_val = df_liq.attrs.get('current_price')
        if isinstance(current_price_val, (int, float)) and current_price_val > 1e-18:
            df_liq['is_near_dynamic'] = (abs(
                df_liq['price'] - current_price_val) / current_price_val * 100) <= proximity_percent
        else:
            df_liq['is_near_dynamic'] = False  # Default if current_price_val is invalid

        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_liq[~df_liq['is_near_dynamic']]['price'],
                                 y=df_liq[~df_liq['is_near_dynamic']]['liquidity_value_approx'],
                                 name='Liquidez Distante', marker_color='lightgrey'))
            fig.add_trace(go.Bar(x=df_liq[df_liq['is_near_dynamic']]['price'],
                                 y=df_liq[df_liq['is_near_dynamic']]['liquidity_value_approx'],
                                 name=f'Liquidez Próxima (±{proximity_percent}%)', marker_color='royalblue'))
            if isinstance(current_price_val, (int, float)): fig.add_vline(x=current_price_val, line_width=2,
                                                                          line_dash="dash", line_color="red",
                                                                          name="Preço Atual")
            fig.update_layout(title=f"Distribuição de Liquidez (Gross) para {df_liq.attrs.get('token_pair', 'Pool')}",
                              xaxis_title=f"Preço ({df_liq.attrs.get('token_pair', 'Tokens')})",
                              yaxis_title=f"Liquidez (Aprox. em {st.session_state.detailed_pool_data.get('token1_symbol', 'T1')})",
                              bargap=0.01, height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Plotly não instalado. Gráfico não pode ser exibido.")
        with st.expander("Ver Dados Tabulares da Liquidez (Gross)"):
            st.dataframe(df_liq[['tick', 'price', 'liquidity_raw', 'liquidity_value_approx', 'distance_percent',
                                 'is_near_dynamic']])
    elif df_liq is None:
        st.warning("Não foi possível carregar dados de liquidez (gross) para esta pool.")
    else:  # df_liq is an empty DataFrame
        st.info(f"Nenhuma informação de liquidez (gross) encontrada para a pool {current_pool_id_for_liq_gross[:8]}...")

with tab3:
    st.header("ℹ️ Explicação dos Métodos de Cálculo")
    st.markdown("""
**Simulador Comparativo de LP Uniswap V3**

Este simulador ajuda a estimar e comparar os potenciais rendimentos de fornecer liquidez (LP) em pools Uniswap V3 em duas estratégias principais:

1.  **Estratégia V2 (Faixa Completa / TVL Share):**
    *   Representa uma posição de LP que cobre toda a faixa de preço (0 a infinito), similar a uma pool V2 tradicional.
    *   O APR e as taxas são calculados com base na sua participação proporcional no TVL total da pool (`Investimento / TVL da Pool`).
    *   Não há concentração de liquidez, então não há "fator de amplificação".

2.  **Estratégia V3 (Liquidez Concentrada Ajustada):**
    *   Permite definir uma faixa de preço específica (`Preço Mínimo` e `Preço Máximo`) para sua liquidez.
    *   **Fator de Amplificação:** Mede o quão mais concentrada sua liquidez está em comparação com uma posição de faixa completa (V2) com o mesmo valor investido. É calculado pela razão entre a liquidez por dólar da sua posição V3 (`L_per_dollar_user_val`) e a liquidez por dólar de uma posição V2 de referência (`L_per_dollar_reference_v2`). Se `L_per_dollar_user_val` for zero ou muito baixo (ex: faixa mal definida), este fator pode ser zero.
    *   **Taxas Potenciais Brutas (Raw Potential):** São as taxas que sua posição V3 geraria se estivesse 100% do tempo dentro da faixa e capturasse 100% das taxas proporcionais à sua liquidez concentrada. Dependem do APR base da pool e do fator de amplificação.
    *   **Tempo na Faixa (TiR - Time in Range):**
        *   Estimativa da porcentagem de tempo que o preço do ativo permanecerá dentro da sua faixa definida.
        *   Pode ser calculado automaticamente com base na volatilidade histórica de 7 dias do par (se disponível e > 0%) ou inserido manualmente.
        *   Se a volatilidade for muito baixa ou indisponível, o TiR manual é usado.
    *   **Eficiência de Captura:**
        *   Representa a porcentagem das taxas (proporcionais à sua liquidez e ao tempo na faixa) que sua posição realmente captura. Fatores como a profundidade da liquidez de outros LPs na sua faixa e a frequência de rebalanceamento podem afetar isso.
        *   **Manual (Slider):** Você define um valor percentual manualmente.
        *   **Baseada em Dados (Beta):** Tenta calcular sua participação na liquidez ativa total da pool no tick de preço atual. Se sua posição estiver fora da faixa de preço atual, ou se os dados de liquidez ativa não puderem ser carregados/processados, ou se a liquidez calculada para o usuário for zero, ou se a eficiência calculada for menor que um limite mínimo (ex: 0.01%), ele reverte para o valor manual.
    *   **APR e Taxas V3 Ajustadas:** São calculadas multiplicando as taxas potenciais brutas pelo fator de TiR e pelo fator de Eficiência de Captura. Se qualquer um desses componentes (taxas brutas, TiR, eficiência) for zero, o resultado final será zero.

**Fontes de Dados:**
*   Os dados das pools (TVL, volume, taxas, preços, ticks) são obtidos via API do The Graph (subgraph oficial da Uniswap V3 para Ethereum Mainnet).
*   O APR base para cálculo pode ser selecionado (último dia, média 7d, média 30d) a partir dos dados históricos da pool. Se o APR base selecionado for 0%, as taxas estimadas V3 também serão 0%.

**Observações:**
*   Todas as estimativas são baseadas em dados históricos e projeções; o desempenho real pode variar.
*   A volatilidade e os volumes de negociação futuros são desconhecidos.
*   A "Eficiência de Captura Baseada em Dados" é uma funcionalidade Beta e pode não ser precisa para todas as pools ou cenários.
""")

logger.info("Aplicação Streamlit renderizada.")

