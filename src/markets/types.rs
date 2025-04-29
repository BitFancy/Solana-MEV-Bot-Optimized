use std::collections::HashMap;
use crate::markets::utils::toPairString;
use serde::{Deserialize, Serialize};
use strum_macros::EnumIter;

#[derive(Debug, Clone, EnumIter, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub enum DexLabel {
    ORCA,
    ORCA_WHIRLPOOLS,
    RAYDIUM,
    RAYDIUM_CLMM,
    METEORA,
}

impl DexLabel {
    pub fn str(&self) -> String {
        match self {
            DexLabel::ORCA => String::from("Orca"),
            DexLabel::ORCA_WHIRLPOOLS => String::from("Orca (Whirlpools)"),
            DexLabel::RAYDIUM => String::from("Raydium"),
            DexLabel::RAYDIUM_CLMM => String::from("Raydium CLMM"),
            DexLabel::METEORA => String::from("Meteora"),
        }
    }
    pub fn api_url(&self) -> String {
        match self {
            DexLabel::ORCA => String::from("https://api.orca.so/allPools"),
            DexLabel::ORCA_WHIRLPOOLS => String::from("https://api.mainnet.orca.so/v1/whirlpool/list"),
            DexLabel::RAYDIUM => String::from("https://api.raydium.io/v2/main/pairs"),
            DexLabel::RAYDIUM_CLMM => String::from("https://api.raydium.io/v2/ammV3/ammPools"),
            DexLabel::METEORA => String::from("https://dlmm-api.meteora.ag/pair/all"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Market {
    pub tokenMintA: String,
    pub tokenVaultA: String,
    pub tokenMintB: String,
    pub tokenVaultB: String,
    pub dexLabel: DexLabel,
    pub fee: u64,
    pub id: String,
    pub account_data: Option<Vec<u8>>,
    pub liquidity: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct Dex {
    pub pairToMarkets: HashMap<String, Vec<Market>>,
    // ammCalcAddPoolMessages: AmmCalcWorkerParamMessage[];
    pub label: DexLabel,
}

impl Dex {
    pub fn new(label: DexLabel) -> Self {
        let pairToMarkets = HashMap::new();
        Dex {
            pairToMarkets: pairToMarkets,
            label: label,
        }
    }
    
    // getAmmCalcAddPoolMessages(): AmmCalcWorkerParamMessage[] {
    //   return this.ammCalcAddPoolMessages;
    // }
    
    pub fn getMarketsForPair(&self, mintA: String, mintB: String) -> &Vec<Market> {
        let pair = toPairString(mintA, mintB);
        let markets = self.pairToMarkets.get(&pair).unwrap();

        return markets;
    }
    
    pub fn getAllMarkets(&self) -> Vec<&Vec<Market>> {
        let mut all_markets = Vec::new();

        for markets in self.pairToMarkets.values() {
            all_markets.push(markets);
        }
        return all_markets;
    }

}

#[derive(Debug)]
pub struct PoolItem {
    pub mintA: String,
    pub mintB: String,
    pub vaultA: String,
    pub vaultB: String,
    pub tradeFeeRate: u128
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SimulationRes {
    pub amountIn: String,
    pub estimatedAmountOut: String,
    pub estimatedMinAmountOut: Option<String>
}
#[derive(Debug, Deserialize, Serialize)]
pub struct SimulationError {
    pub error: String,
}