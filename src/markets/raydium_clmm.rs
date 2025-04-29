use crate::markets::types::{Dex, DexLabel, Market, PoolItem};
use crate::markets::utils::toPairString;
use tokio::net::TcpStream;
use std::{fs::File, io::Read};
use std::fs;
use std::io;
use std::io::Write;
use serde::{Deserialize, Serialize};
use reqwest::get;
use log::{info, error};
use solana_account_decoder::{UiAccountData, UiAccountEncoding};
use solana_program::pubkey::Pubkey;
use solana_sdk::commitment_config::CommitmentConfig;
use solana_pubsub_client::pubsub_client::PubsubClient;
use anyhow::Result;
use solana_client::rpc_client::RpcClient;
use solana_client::rpc_config::RpcAccountInfoConfig;

use crate::common::constants::Env;

#[derive(Debug)]
pub struct RaydiumClmmDEX {
    pub dex: Dex,
    pub pools: Vec<PoolItem>,
}
impl RaydiumClmmDEX {
    pub fn new(mut dex: Dex) -> Self {

        let mut pools_vec = Vec::new();
        
        let data = fs::read_to_string("src\\markets\\cache\\raydiumclmm-markets.json").expect("Error reading file");
        let json_value: Root  = serde_json::from_str(&data).unwrap();

        for pool in json_value.data.clone() {
            let item: PoolItem = PoolItem {
                mintA: pool.mint_a.clone(),
                mintB: pool.mint_b.clone(),
                vaultA: pool.vault_a.clone(),
                vaultB: pool.vault_b.clone(),
                tradeFeeRate: pool.amm_config.trade_fee_rate.clone() as u128,
            };
            pools_vec.push(item);

            let market: Market = Market {
                tokenMintA: pool.mint_a.clone(),
                tokenVaultA: pool.vault_a.clone(),
                tokenMintB: pool.mint_b.clone(),
                tokenVaultB: pool.vault_b.clone(),
                dexLabel: DexLabel::RAYDIUM_CLMM,
                fee: pool.amm_config.trade_fee_rate.clone() as u64,
                id: pool.id.clone(),
                account_data: None,
                liquidity: None,
            };

            let pair_string = toPairString(pool.mint_a, pool.mint_b);
            if dex.pairToMarkets.contains_key(&pair_string.clone()) {
                let vec_market = dex.pairToMarkets.get_mut(&pair_string).unwrap();
                vec_market.push(market);
            } else {
                dex.pairToMarkets.insert(pair_string, vec![market]);
            }
        }

        info!("Raydium CLMM: {} pools founded", json_value.data.len());
        Self {
            dex: dex,
            pools: pools_vec,
        }
    }
  }

pub async fn fetch_data_raydium_clmm() -> Result<(), Box<dyn std::error::Error>> {
    let response = get("https://api.raydium.io/v2/ammV3/ammPools").await?;
    // info!("response: {:?}", response);
    // info!("response-status: {:?}", response.status().is_success());
    if response.status().is_success() {
        let json: Root = serde_json::from_str(&response.text().await?)?;        
        // info!("json: {:?}", json);
        let mut file = File::create("src\\markets\\cache\\raydiumclmm-markets.json")?;
        file.write_all(serde_json::to_string(&json)?.as_bytes())?;
        info!("Data written to 'raydiumclmm-markets.json' successfully.");
    } else {
        error!("Fetch of 'raydiumclmm-markets.json' not successful: {}", response.status());
    }
    Ok(())
}

pub async fn stream_raydium_clmm(account: Pubkey) -> Result<()> {
    let env = Env::new();
    let url = env.wss_rpc_url.as_str();
    let (mut account_subscription_client, account_subscription_receiver) =
    PubsubClient::account_subscribe(
        url,
        &account,
        Some(RpcAccountInfoConfig {
            encoding: Some(UiAccountEncoding::JsonParsed),
            data_slice: None,
            commitment: Some(CommitmentConfig::confirmed()),
            min_context_slot: None,
        }),
    )?;

    loop {
        match account_subscription_receiver.recv() {
            Ok(response) => {
                let data = response.value.data;
                let bytes_slice = UiAccountData::decode(&data).unwrap();
                println!("account subscription data response: {:?}", data);
                // let account_data = unpack_from_slice(bytes_slice.as_slice());
                // println!("Raydium CLMM Pool updated: {:?}", account);
                // println!("Data: {:?}", account_data.unwrap());

            }
            Err(e) => {
                error!("account subscription error: {:?}", e);
                break;
            }
        }
    }

    Ok(())
}

// // Simulate one route 
// pub async fn simulate_route_raydium_clmm(amount_in: f64, route: Route, market: Market, tokens_infos: HashMap<String, TokenInfos>) -> String {
//     // I want to get the data of the market i'm interested in this route
//     let bytes_slice = UiAccountData::try_from_slice(market.account_data);
//     let mut decimals_0: u8 = 0;
//     let mut decimals_1: u8 = 0;
    
//     // if route.token_0to1 == true {
//         decimals_0 = tokens_infos.get(&market.tokenMintA).unwrap().decimals;
//         decimals_1 = tokens_infos.get(&market.tokenMintB).unwrap().decimals;
//     // } else {
//     //     decimals_0 = tokens_infos.get(&market.tokenMintB).unwrap().decimals;
//     //     decimals_1 = tokens_infos.get(&market.tokenMintA).unwrap().decimals; 
//     // }

//     //Get price
//     // let price = from_x64_orca_wp(whirpool_data.sqrt_price, decimals_0 as f64, decimals_1 as f64);
//     // println!("Price: {:?}", price);

//     // Simulate a swap
//     let env = Env::new();
//     let domain = env.simulator_url;

//     let params = format!(
//         "tokenInKey={}&tokenInDecimals={}&tokenOutKey={}&tokenOutDecimals={}&tickSpacing={}&amountIn={}",
//         whirpool_data.token_mint_a,
//         decimals_0,
//         whirpool_data.token_mint_b,
//         decimals_1,
//         whirpool_data.tick_spacing,
//         amount_in,
//     );
//     let req_url = format!("{}orca_quote?{}", domain, params);
//     // println!("req_url: {:?}", req_url);

//     let mut res = reqwest::get(req_url).await.expect("Error in request to simulator");

//     let json_value = res.json::<SimulationRes>().await;
//     match json_value {
//         Ok(value) => {
//             println!("estimatedAmountIn: {:?}", value.estimatedAmountIn);
//             println!("estimatedAmountOut: {:?}", value.estimatedAmountOut);
//             return value.estimatedAmountOut;
//         }
//         Err(value) => { format!("value{:?}", value) }
//     }

// }



#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Root {
    pub data: Vec<RaydiumCLmmPool>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RaydiumCLmmPool {
    pub id: String,
    pub mint_program_id_a: String,
    pub mint_program_id_b: String,
    pub mint_a: String,
    pub mint_b: String,
    pub vault_a: String,
    pub vault_b: String,
    pub mint_decimals_a: i64,
    pub mint_decimals_b: i64,
    pub amm_config: AmmConfig,
    pub reward_infos: Vec<RewardInfo>,
    pub tvl: f64,
    pub day: Day,
    pub week: Week,
    pub month: Month,
    pub lookup_table_account: Option<String>,
    pub open_time: i64,
    pub price: f64,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AmmConfig {
    pub id: String,
    pub index: i64,
    pub protocol_fee_rate: i64,
    pub trade_fee_rate: i64,
    pub tick_spacing: i64,
    pub fund_fee_rate: i64,
    pub fund_owner: String,
    pub description: String,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RewardInfo {
    pub mint: String,
    pub program_id: String,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Day {
    pub volume: f64,
    pub volume_fee: f64,
    pub fee_a: f64,
    pub fee_b: f64,
    pub fee_apr: f64,
    pub reward_apr: RewardApr,
    pub apr: f64,
    pub price_min: f64,
    pub price_max: f64,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RewardApr {
    #[serde(rename = "A")]
    pub a: f64,
    #[serde(rename = "B")]
    pub b: f64,
    #[serde(rename = "C")]
    pub c: i64,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Week {
    pub volume: f64,
    pub volume_fee: f64,
    pub fee_a: f64,
    pub fee_b: f64,
    pub fee_apr: f64,
    pub reward_apr: RewardApr2,
    pub apr: f64,
    pub price_min: f64,
    pub price_max: f64,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RewardApr2 {
    #[serde(rename = "A")]
    pub a: f64,
    #[serde(rename = "B")]
    pub b: f64,
    #[serde(rename = "C")]
    pub c: i64,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Month {
    pub volume: f64,
    pub volume_fee: f64,
    pub fee_a: f64,
    pub fee_b: f64,
    pub fee_apr: f64,
    pub reward_apr: RewardApr3,
    pub apr: f64,
    pub price_min: f64,
    pub price_max: f64,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RewardApr3 {
    #[serde(rename = "A")]
    pub a: f64,
    #[serde(rename = "B")]
    pub b: f64,
    #[serde(rename = "C")]
    pub c: i64,
}
