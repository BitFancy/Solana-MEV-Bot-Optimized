use crate::common::constants::Env;
use crate::markets::types::{Dex, DexLabel, Market, PoolItem};
use crate::markets::utils::toPairString;
use crate::common::utils::{from_str, from_Pubkey};
use std::collections::HashMap;
use std::{fs, fs::File};
use std::io::Write;
use serde::{Deserialize, Serialize};
use reqwest::get;
use log::info;
use solana_client::rpc_client::RpcClient;
use solana_client::rpc_config::RpcAccountInfoConfig;
use solana_program::pubkey::Pubkey;
use solana_sdk::commitment_config::CommitmentConfig;
use solana_sdk::program_error::ProgramError;
use solana_account_decoder::{UiAccountData, UiAccountEncoding};
use solana_pubsub_client::pubsub_client::PubsubClient;
use anyhow::Result;

#[derive(Debug)]
pub struct OrcaDex {
    pub dex: Dex,
    pub pools: Vec<PoolItem>,
}
impl OrcaDex {
    pub fn new(mut dex: Dex) -> Self {

        let env = Env::new();
        let rpc_client = RpcClient::new(env.rpc_url);

        let mut pools_vec = Vec::new();
        
        let data = fs::read_to_string("src\\markets\\cache\\orca-markets.json").expect("Error reading file");
        let json_value: HashMap<String, Pool>  = serde_json::from_str(&data).unwrap();

        // println!("JSON Pools: {:?}", json_value);

        let mut pubkeys_vec: Vec<Pubkey> = Vec::new();

        for (key, pool) in json_value.clone() {
            let pubkey = from_str(pool.pool_account.as_str()).unwrap();
            pubkeys_vec.push(pubkey);
        }

        let mut results_pools = Vec::new();

        for i in (0..pubkeys_vec.len()).step_by(100) {
            let maxLength = std::cmp::min(i + 100, pubkeys_vec.len());
            let batch = &pubkeys_vec[(i..maxLength)];

            let batch_results = rpc_client.get_multiple_accounts(&batch).unwrap();
            for j in batch_results {
                let account = j.unwrap();
                let data = unpack_from_slice(&account.data.into_boxed_slice());
                results_pools.push(data.unwrap());
            }
        }

        for pool in &results_pools {

            let fee = (pool.trade_fee_numerator as f64 / pool.trade_fee_denominator as f64) * 10000 as f64;

            let item: PoolItem = PoolItem {
                mintA: from_Pubkey(pool.mint_a.clone()),
                mintB: from_Pubkey(pool.mint_b.clone()),
                vaultA: from_Pubkey(pool.token_account_a.clone()),
                vaultB: from_Pubkey(pool.token_account_b.clone()),
                tradeFeeRate: fee.clone() as u128,
            };

            pools_vec.push(item);

            let market: Market = Market {
                tokenMintA: from_Pubkey(pool.mint_a.clone()),
                tokenVaultA: from_Pubkey(pool.token_account_a.clone()),
                tokenMintB: from_Pubkey(pool.mint_b.clone()),
                tokenVaultB: from_Pubkey(pool.token_account_b.clone()),
                fee: fee.clone() as u64,
                dexLabel: DexLabel::ORCA,
                id: from_Pubkey(pool.token_pool.clone()),
                account_data: None,
                liquidity: None,
            };

            let pair_string = toPairString(from_Pubkey(pool.mint_a), from_Pubkey(pool.mint_b));
            if dex.pairToMarkets.contains_key(&pair_string.clone()) {
                let vec_market = dex.pairToMarkets.get_mut(&pair_string).unwrap();
                vec_market.push(market);
            } else {
                dex.pairToMarkets.insert(pair_string, vec![market]);
            }
        }

        info!("Orca: {} pools founded", &results_pools.len());
        Self {
            dex: dex,
            pools: pools_vec,
        }
    }
  }

pub async fn fetch_data_orca() -> Result<(), Box<dyn std::error::Error>> {
    let response = get("https://api.orca.so/allPools").await?;
    // info!("response: {:?}", response);
    // info!("response-status: {:?}", response.status().is_success());
    if response.status().is_success() {
        let json: HashMap<String, Pool> = serde_json::from_str(&response.text().await?)?;        
        // info!("json: {:?}", json);
        let mut file = File::create("src\\markets\\cache\\orca-markets.json")?;
        file.write_all(serde_json::to_string(&json)?.as_bytes())?;
        info!("Data written to 'orca-markets.json' successfully.");
    } else {
        info!("Fetch of 'orca-markets.json' not successful: {}", response.status());
    }
    Ok(())
}

pub async fn stream_orca(account: Pubkey) -> Result<()> {
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
                // println!("account subscription data response: {:?}", data);
                let account_data = unpack_from_slice(bytes_slice.as_slice());
                println!("Orca Pool updated: {:?}", account);
                println!("Data: {:?}", account_data.unwrap());

            }
            Err(e) => {
                println!("account subscription error: {:?}", e);
                break;
            }
        }
    }

    Ok(())
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Root {
    pool: Pool,
} 

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Pool {
    pub pool_id: String,
    pub pool_account: String,
    #[serde(rename = "tokenAAmount")]
    pub token_aamount: String,
    #[serde(rename = "tokenBAmount")]
    pub token_bamount: String,
    pub pool_token_supply: String,
    pub apy: Apy,
    pub volume: Volume,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Apy {
    pub day: String,
    pub week: String,
    pub month: String,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Volume {
    pub day: String,
    pub week: String,
    pub month: String,
}

#[derive(Debug)]
pub struct TokenSwapLayout {
    pub version: u8,
    pub is_initialized: bool,
    pub bump_seed: u8,
    pub pool_token_program_id: Pubkey,
    pub token_account_a: Pubkey,
    pub token_account_b: Pubkey,
    pub token_pool: Pubkey,
    pub mint_a: Pubkey,
    pub mint_b: Pubkey,
    pub fee_account: Pubkey,
    pub trade_fee_numerator: u64,
    pub trade_fee_denominator: u64,
    pub owner_trade_fee_numerator: u64,
    pub owner_trade_fee_denominator: u64,
    pub owner_withdraw_fee_numerator: u64,
    pub owner_withdraw_fee_denominator: u64,
    pub host_fee_numerator: u64,
    pub host_fee_denominator: u64,
    pub curve_type: u8,
    pub curve_parameters: [u8; 32],
}

fn unpack_from_slice(src: &[u8]) -> Result<TokenSwapLayout, ProgramError> {
    let version = src[0];
    let is_initialized = src[1] != 0;
    let bump_seed = src[2];
    let pool_token_program_id = Pubkey::new_from_array(<[u8; 32]>::try_from(&src[3..35]).expect("Orca pools bad unpack"));
    let token_account_a = Pubkey::new_from_array(<[u8; 32]>::try_from(&src[35..67]).expect("Orca pools bad unpack"));
    let token_account_b = Pubkey::new_from_array(<[u8; 32]>::try_from(&src[67..99]).expect("Orca pools bad unpack"));
    let token_pool = Pubkey::new_from_array(<[u8; 32]>::try_from(&src[99..131]).expect("Orca pools bad unpack"));
    let mint_a = Pubkey::new_from_array(<[u8; 32]>::try_from(&src[131..163]).expect("Orca pools bad unpack"));
    let mint_b = Pubkey::new_from_array(<[u8; 32]>::try_from(&src[163..195]).expect("Orca pools bad unpack"));
    let fee_account = Pubkey::new_from_array(<[u8; 32]>::try_from(&src[195..227]).expect("Orca pools bad unpack"));
    let trade_fee_numerator = u64::from_le_bytes(<[u8; 8]>::try_from(&src[227..235]).expect("Orca pools bad unpack"));
    let trade_fee_denominator = u64::from_le_bytes(<[u8; 8]>::try_from(&src[235..243]).expect("Orca pools bad unpack"));
    let owner_trade_fee_numerator = u64::from_le_bytes(<[u8; 8]>::try_from(&src[243..251]).expect("Orca pools bad unpack"));
    let owner_trade_fee_denominator = u64::from_le_bytes(<[u8; 8]>::try_from(&src[251..259]).expect("Orca pools bad unpack"));
    let owner_withdraw_fee_numerator = u64::from_le_bytes(<[u8; 8]>::try_from(&src[259..267]).expect("Orca pools bad unpack"));
    let owner_withdraw_fee_denominator = u64::from_le_bytes(<[u8; 8]>::try_from(&src[267..275]).expect("Orca pools bad unpack"));
    let host_fee_numerator = u64::from_le_bytes(<[u8; 8]>::try_from(&src[275..283]).expect("Orca pools bad unpack"));
    let host_fee_denominator = u64::from_le_bytes(<[u8; 8]>::try_from(&src[283..291]).expect("Orca pools bad unpack"));
    let curve_type = src[291];
    let mut curve_parameters = [0u8; 32];
    curve_parameters.copy_from_slice(&src[292..]);

    Ok(TokenSwapLayout {
        version,
        is_initialized,
        bump_seed,
        pool_token_program_id,
        token_account_a,
        token_account_b,
        token_pool,
        mint_a,
        mint_b,
        fee_account,
        trade_fee_numerator,
        trade_fee_denominator,
        owner_trade_fee_numerator,
        owner_trade_fee_denominator,
        owner_withdraw_fee_numerator,
        owner_withdraw_fee_denominator,
        host_fee_numerator,
        host_fee_denominator,
        curve_type,
        curve_parameters,
    })
}
