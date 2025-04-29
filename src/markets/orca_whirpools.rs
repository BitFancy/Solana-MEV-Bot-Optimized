use crate::arbitrage::types::{Route, TokenInfos};
use crate::common::constants::Env;
use crate::markets::types::{Dex, DexLabel, Market, PoolItem, SimulationRes};
use crate::markets::utils::toPairString;
use crate::common::utils::{from_Pubkey, from_str, make_request};
use std::collections::HashMap;
use std::{fs, fs::File};
use std::io::Write;
use borsh::{BorshDeserialize, BorshSerialize};
use serde::{Deserialize, Serialize};
use reqwest::get;
use log::{info, error};
use solana_account_decoder::{UiAccountData, UiAccountEncoding};
use solana_client::rpc_client::RpcClient;
use solana_client::rpc_config::{RpcAccountInfoConfig, RpcProgramAccountsConfig};
use solana_client::rpc_filter::{Memcmp, MemcmpEncodedBytes, RpcFilterType};
use solana_program::pubkey::Pubkey;
use solana_sdk::commitment_config::CommitmentConfig;
use solana_sdk::program_error::ProgramError;
use solana_pubsub_client::pubsub_client::PubsubClient;
use anyhow::Result;

use super::types::SimulationError;

#[derive(Debug)]
pub struct OrcaDexWhirpools {
    pub dex: Dex,
    pub pools: Vec<PoolItem>,
}
impl OrcaDexWhirpools {
    pub fn new(mut dex: Dex) -> Self {

        let env = Env::new();
        let rpc_client = RpcClient::new(env.rpc_url);

        let mut pools_vec = Vec::new();
        
        let data = fs::read_to_string("src\\markets\\cache\\orca_whirpools-markets.json").expect("Error reading file");
        let json_value: Root = serde_json::from_str(&data).unwrap();

        // println!("JSON Pools: {:?}", json_value.whirlpools);

        let mut pubkeys_vec: Vec<Pubkey> = Vec::new();

        for pool in json_value.whirlpools.clone() {
            let pubkey = from_str(pool.address.as_str()).unwrap();
            pubkeys_vec.push(pubkey);
        }
        
        let mut results_pools = Vec::new();
        
        for i in (0..pubkeys_vec.len()).step_by(100) {
            let maxLength = std::cmp::min(i + 100, pubkeys_vec.len());
            let batch = &pubkeys_vec[(i..maxLength)];
            
            let batch_results = rpc_client.get_multiple_accounts(&batch).unwrap();
            for (j, account) in batch_results.iter().enumerate() {
                let account = account.clone().unwrap();
                // let gov = solana_sdk::pubkey::Pubkey::try_from("2LecshUwdy9xi7meFgHtFJQNSKk4KdTrcpvaB56dP2NQ").unwrap();
                // println!("Gov bytes: {:?}", gov.to_bytes());
                // println!("Account {:?}", &account);
                let mut data = unpack_from_slice(&account.data).unwrap();
                // println!("Data {:?}", &data);

                data.address = batch[j];
                // println!("WhirpoolAccount: {:?}", data);
                results_pools.push(data);
            }
        }
            
        // println!("Print Whirpool Account {:?}", &results_pools[76]);
        // println!("Print Whirpool Account {:?}", &results_pools[162]);
        // println!("Print Whirpool Account {:?}", &results_pools[3726]);

        for pool in &results_pools {

            // let fee = (pool.trade_fee_numerator as f64 / pool.trade_fee_denominator as f64) * 10000 as f64;

            let item: PoolItem = PoolItem {
                mintA: from_Pubkey(pool.token_mint_a.clone()),
                mintB: from_Pubkey(pool.token_mint_b.clone()),
                vaultA: from_Pubkey(pool.token_vault_a.clone()),
                vaultB: from_Pubkey(pool.token_vault_b.clone()),
                tradeFeeRate: pool.fee_rate.clone() as u128,
            };

            pools_vec.push(item);

            let market: Market = Market {
                tokenMintA: from_Pubkey(pool.token_mint_a.clone()),
                tokenVaultA: from_Pubkey(pool.token_vault_a.clone()),
                tokenMintB: from_Pubkey(pool.token_mint_b.clone()),
                tokenVaultB: from_Pubkey(pool.token_vault_b.clone()),
                fee: pool.fee_rate.clone() as u64,
                dexLabel: DexLabel::ORCA_WHIRLPOOLS,
                id: from_Pubkey(pool.address.clone()),
                //TODO: None here, be sure to refresh data after
                account_data: None, 
                liquidity: Some(pool.liquidity as u64),
            };

            let pair_string = toPairString(from_Pubkey(pool.token_mint_a), from_Pubkey(pool.token_mint_b));
            if dex.pairToMarkets.contains_key(&pair_string.clone()) {
                let vec_market = dex.pairToMarkets.get_mut(&pair_string).unwrap();
                vec_market.push(market);
            } else {
                dex.pairToMarkets.insert(pair_string, vec![market]);
            }
        }

        info!("Orca Whirpools: {} pools founded", &results_pools.len());
        Self {
            dex: dex,
            pools: pools_vec,
        }
    }
}

pub async fn fetch_new_orca_whirpools(rpc_client: &RpcClient, token: String, on_tokena: bool) -> Vec<(Pubkey, Market)> {
    let orca_program = "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc".to_string();
    
    let mut new_markets: Vec<(Pubkey, Market)> = Vec::new(); 
    let filters = Some(vec![
        RpcFilterType::Memcmp(Memcmp::new(
            // 101 for token_mint_a && 181 for token_mint_b
            if on_tokena == true {
                101
            } else {
                181
            }, 
          MemcmpEncodedBytes::Base58(token.clone()),
        )),
        RpcFilterType::DataSize(653),  //data.len == 653 for Whirpool account
    ]);
    
    let accounts = rpc_client.get_program_accounts_with_config(
        &from_str(&orca_program).unwrap(),
        RpcProgramAccountsConfig {
            filters,
            account_config: RpcAccountInfoConfig {
                encoding: Some(UiAccountEncoding::Base64),
                commitment: Some(rpc_client.commitment()),
                ..RpcAccountInfoConfig::default()
            },
            ..RpcProgramAccountsConfig::default()
        },
    ).unwrap();

    for account in accounts {
        let whirpool_account = unpack_from_slice(account.1.data.as_slice()).unwrap();
        let market: Market = Market {
            tokenMintA: from_Pubkey(whirpool_account.token_mint_a.clone()),
            tokenVaultA: from_Pubkey(whirpool_account.token_vault_a.clone()),
            tokenMintB: from_Pubkey(whirpool_account.token_mint_b.clone()),
            tokenVaultB: from_Pubkey(whirpool_account.token_vault_b.clone()),
            fee: whirpool_account.fee_rate.clone() as u64,
            dexLabel: DexLabel::ORCA_WHIRLPOOLS,
            id: from_Pubkey(account.0.clone()),
            account_data: Some(account.1.data),
            liquidity: Some(whirpool_account.liquidity as u64),
        };
        new_markets.push((account.0, market));
    }
    // println!("Accounts: {:?}", accounts);
    return new_markets;
}

pub async fn fetch_data_orca_whirpools() -> Result<(), Box<dyn std::error::Error>> {
    let response = get("https://api.mainnet.orca.so/v1/whirlpool/list").await?;
    // info!("response: {:?}", response);
    // info!("response-status: {:?}", response.status().is_success());
    if response.status().is_success() {
        let json: Root = serde_json::from_str(&response.text().await?)?;        
        // info!("json: {:?}", json);
        let mut file = File::create("src\\markets\\cache\\orca_whirpools-markets.json")?;
        file.write_all(serde_json::to_string(&json)?.as_bytes())?;
        info!("Data written to 'orca_whirpools-markets.json' successfully.");
    } else {
        error!("Fetch of 'orca_whirpools-markets.json' not successful: {}", response.status());
    }
    Ok(())
}

pub async fn stream_orca_whirpools(account: Pubkey) -> Result<()> {
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
                println!("Orca Whirpools Pool updated: {:?}", account);
                println!("Data: {:?}", account_data.unwrap());

            }
            Err(e) => {
                error!("account subscription error: {:?}", e);
                break;
            }
        }
    }

    Ok(())
}

// Simulate one route 
pub async fn simulate_route_orca_whirpools(printing_amt: bool, amount_in: u64, route: Route, market: Market, tokens_infos: HashMap<String, TokenInfos>) -> Result<(String, String), Box<dyn std::error::Error>> {
    // I want to get the data of the market i'm interested in this route
    let whirpool_data = unpack_from_slice(market.account_data.expect("No account data provided").as_slice()).unwrap();

    let token_0 = tokens_infos.get(&market.tokenMintA).unwrap();
    let token_1 = tokens_infos.get(&market.tokenMintB).unwrap();
    let mut params: String = String::new();
    if route.token_0to1 {
        params = format!(
            "poolId={}&tokenInKey={}&tokenInDecimals={}&tokenInSymbol={}&tokenOutKey={}&tokenOutDecimals={}&tokenOutSymbol={}&tickSpacing={}&amountIn={}",
            route.pool_address,
            whirpool_data.token_mint_a,
            token_0.decimals,
            token_0.symbol,
            whirpool_data.token_mint_b,
            token_1.decimals,
            token_1.symbol,
            whirpool_data.tick_spacing,
            amount_in,
        );
    } else {
        params = format!(
            "poolId={}&tokenInKey={}&tokenInDecimals={}&tokenInSymbol={}&tokenOutKey={}&tokenOutDecimals={}&tokenOutSymbol={}&tickSpacing={}&amountIn={}",
            route.pool_address,
            whirpool_data.token_mint_b,
            token_1.decimals,
            token_1.symbol,
            whirpool_data.token_mint_a,
            token_0.decimals,
            token_0.symbol,
            whirpool_data.tick_spacing,
            amount_in,
        );
    }

    //Get price
    // let price = from_x64_orca_wp(whirpool_data.sqrt_price, decimals_0 as f64, decimals_1 as f64);
    // println!("Price: {:?}", price);

    // Simulate a swap
    let env = Env::new();
    let domain = env.simulator_url;

    let req_url = format!("{}orca_quote?{}", domain, params);
    // println!("req_url: {:?}", req_url);

    let res = make_request(req_url).await?;
    let res_text = res.text().await?;

    if let Ok(json_value) = serde_json::from_str::<SimulationRes>(&res_text) {
        if printing_amt {
            println!("estimatedAmountIn: {:?} {:?}", json_value.amountIn, token_0.symbol);
            println!("estimatedAmountOut: {:?} {:?}", json_value.estimatedAmountOut, token_1.symbol);
            println!("estimatedMinAmountOut: {:?} {:?}", json_value.estimatedMinAmountOut.clone().unwrap(), token_1.symbol);
        }
        
        Ok((json_value.estimatedAmountOut, json_value.estimatedMinAmountOut.unwrap_or_default()))
    } else if let Ok(error_value) = serde_json::from_str::<SimulationError>(&res_text) {
        // println!("ERROR Value: {:?}", error_value.error);
        Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            error_value.error,
        )))
    } else {
        Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Unexpected response format",
        )))
    }
}

// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// :::::::::::::::::::::::::::::::::::::                      UTILS                   :::::::::::::::::::::::::::::::::::::::::::::
// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

pub fn get_price() {

}

pub fn unpack_from_slice(src: &[u8]) -> Result<WhirlpoolAccount, ProgramError> {
    let address = Pubkey::new_from_array([0 as u8; 32]);    // Init to 0 and update just after
    let whirlpools_config = Pubkey::new_from_array(<[u8; 32]>::try_from(&src[8..40]).expect("Orca pools bad unpack"));
    let whirlpool_bump = [src[40]];
    let tick_spacing = u16::from_le_bytes(<[u8; 2]>::try_from(&src[41..43]).expect("Orca pools bad unpack"));
    let tick_spacing_seed = [src[43], src[44]];
    let fee_rate = u16::from_le_bytes(<[u8; 2]>::try_from(&src[45..47]).expect("Orca pools bad unpack"));
    let protocol_fee_rate = u16::from_le_bytes(<[u8; 2]>::try_from(&src[47..49]).expect("Orca pools bad unpack"));
    let liquidity = u128::from_le_bytes(<[u8; 16]>::try_from(&src[49..65]).expect("Orca pools bad unpack"));
    let sqrt_price = u128::from_le_bytes(<[u8; 16]>::try_from(&src[65..81]).expect("Orca pools bad unpack"));
    let tick_current_index = i32::from_le_bytes(<[u8; 4]>::try_from(&src[81..85]).expect("Orca pools bad unpack"));
    let protocol_fee_owed_a = u64::from_le_bytes(<[u8; 8]>::try_from(&src[85..93]).expect("Orca pools bad unpack"));
    let protocol_fee_owed_b = u64::from_le_bytes(<[u8; 8]>::try_from(&src[93..101]).expect("Orca pools bad unpack"));
    let token_mint_a = Pubkey::new_from_array(<[u8; 32]>::try_from(&src[101..133]).expect("Orca pools bad unpack"));
    let token_vault_a = Pubkey::new_from_array(<[u8; 32]>::try_from(&src[133..165]).expect("Orca pools bad unpack"));
    let fee_growth_global_a = u128::from_le_bytes(<[u8; 16]>::try_from(&src[165..181]).expect("Orca pools bad unpack"));
    let token_mint_b = Pubkey::new_from_array(<[u8; 32]>::try_from(&src[181..213]).expect("Orca pools bad unpack"));
    let token_vault_b = Pubkey::new_from_array(<[u8; 32]>::try_from(&src[213..245]).expect("Orca pools bad unpack"));
    let fee_growth_global_b = u128::from_le_bytes(<[u8; 16]>::try_from(&src[245..261]).expect("Orca pools bad unpack"));   
    let reward_last_updated_timestamp = u64::from_le_bytes(<[u8; 8]>::try_from(&src[261..269]).expect("Orca pools bad unpack"));
    // let reward_infos = 

    Ok(WhirlpoolAccount {
        address,
        whirlpools_config,
        whirlpool_bump,
        tick_spacing,
        tick_spacing_seed,
        fee_rate,
        protocol_fee_rate,
        liquidity,
        sqrt_price,
        tick_current_index,
        protocol_fee_owed_a,
        protocol_fee_owed_b,
        token_mint_a,
        token_vault_a,
        fee_growth_global_a,
        token_mint_b,
        token_vault_b,
        fee_growth_global_b,
        reward_last_updated_timestamp,
    })
}


// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// :::::::::::::::::::::::::::::::::::::                      TYPES                   :::::::::::::::::::::::::::::::::::::::::::::
// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Root {
    pub whirlpools: Vec<Whirlpool>,
    pub has_more: bool,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Whirlpool {
    pub address: String,
    pub token_a: Token,
    pub token_b: Token,
    pub whitelisted: bool,
    pub tick_spacing: i64,
    pub price: f64,
    pub lp_fee_rate: f64,
    pub protocol_fee_rate: f64,
    pub whirlpools_config: String,
    pub modified_time_ms: Option<i64>,
    pub tvl: Option<f64>,
    pub volume: Option<RewardApr>,
    pub volume_denominated_a: Option<RewardApr>,
    pub volume_denominated_b: Option<RewardApr>,
    pub price_range: Option<PriceRange>,
    pub fee_apr: Option<RewardApr>,
    #[serde(rename = "reward0Apr")]
    pub reward0apr: Option<RewardApr>,
    #[serde(rename = "reward1Apr")]
    pub reward1apr: Option<RewardApr>,
    #[serde(rename = "reward2Apr")]
    pub reward2apr: Option<RewardApr>,
    pub total_apr: Option<RewardApr>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Token {
    pub mint: String,
    pub symbol: String,
    pub name: String,
    pub decimals: i64,
    #[serde(rename = "logoURI")]
    pub logo_uri: Option<String>,
    pub coingecko_id: Option<String>,
    pub whitelisted: bool,
    pub pool_token: bool,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PriceRange {
    pub day: Time,
    pub week: Time,
    pub month: Time,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Time {
    pub min: f64,
    pub max: f64,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RewardApr {
    pub day: Option<f64>,
    pub week: Option<f64>,
    pub month: Option<f64>,
}

#[derive(Debug, Clone, BorshDeserialize, BorshSerialize)]
pub struct WhirlpoolAccount {
    pub address: Pubkey,
    pub whirlpools_config: Pubkey,
    pub whirlpool_bump: [u8; 1],
    pub tick_spacing: u16,
    pub tick_spacing_seed: [u8; 2],
    pub fee_rate: u16,
    pub protocol_fee_rate: u16,
    pub liquidity: u128,
    pub sqrt_price: u128,
    pub tick_current_index: i32,
    pub protocol_fee_owed_a: u64,
    pub protocol_fee_owed_b: u64,
    pub token_mint_a: Pubkey,
    pub token_vault_a: Pubkey,
    pub fee_growth_global_a: u128,
    pub token_mint_b: Pubkey,
    pub token_vault_b: Pubkey,
    pub fee_growth_global_b: u128,
    pub reward_last_updated_timestamp: u64,
}

#[derive(Debug, Clone, BorshDeserialize, BorshSerialize)]
pub struct WhirlpoolAccountRewardInfo {
    pub mint: Pubkey,
    pub vault: Pubkey,
    pub authority: Pubkey,
    pub emissions_per_second_x64: u128,
    pub growth_global_x64: u128,
}

#[derive(Debug, Clone, BorshDeserialize, BorshSerialize)]
pub struct WhirlpoolAccountState {
    pub sig: u64,
    pub whirlpools_config: Pubkey,
    pub whirlpool_bump: [u8; 1],
    pub tick_spacing: u16,
    pub tick_spacing_seed: [u8; 2],
    pub fee_rate: u16,
    pub protocol_fee_rate: u16,
    pub liquidity: u128,
    pub sqrt_price: u128,
    pub tick_current_index: i32,
    pub protocol_fee_owed_a: u64,
    pub protocol_fee_owed_b: u64,
    pub token_mint_a: Pubkey,
    pub token_vault_a: Pubkey,
    pub fee_growth_global_a: u128,
    pub token_mint_b: Pubkey,
    pub token_vault_b: Pubkey,
    pub fee_growth_global_b: u128,
    pub reward_last_updated_timestamp: u64,
    pub reward_infos: [WhirlpoolAccountRewardInfo; 3],
}