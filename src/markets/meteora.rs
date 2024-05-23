use crate::arbitrage::types::{Route, TokenInfos};
use crate::markets::types::{Dex, DexLabel, Market, PoolItem, SimulationRes};
use crate::markets::utils::toPairString;
use crate::common::debug::print_json_segment;
use crate::common::utils::{from_Pubkey, from_str, make_request};
use crate::common::constants::Env;

use borsh::{BorshDeserialize, BorshSerialize};
use eth_encode_packed::ethabi::token;
use solana_client::rpc_filter::{Memcmp, MemcmpEncodedBytes, RpcFilterType};
use tokio::net::TcpStream;
use std::collections::HashMap;
use std::{fs::File, io::Read};
use std::fs;
use serde::{Deserialize, Deserializer, de, Serialize};
use serde_json::Value;
use reqwest::get;
use std::io::{BufWriter, Write};
use futures::StreamExt;
use log::{info, error};
use solana_account_decoder::{UiAccountData, UiAccountEncoding};
use solana_program::pubkey::Pubkey;
use solana_sdk::commitment_config::CommitmentConfig;
use solana_pubsub_client::pubsub_client::PubsubClient;
use anyhow::Result;
use solana_client::rpc_client::RpcClient;
use solana_client::rpc_config::{RpcAccountInfoConfig, RpcProgramAccountsConfig};

#[derive(Debug)]
pub struct MeteoraDEX {
    pub dex: Dex,
    pub pools: Vec<PoolItem>,
}
impl MeteoraDEX {
    pub fn new(mut dex: Dex) -> Self {

        let mut pools_vec = Vec::new();
        
        let data = fs::read_to_string("src\\markets\\cache\\meteora-markets.json").expect("LogRocket: error reading file");
        let json_value: Root = serde_json::from_str(&data).unwrap();

        
        for pool in json_value.clone() {
            //Serialization foraccount_data
            let mut serialized_data: Vec<u8> = Vec::new();
            let result = BorshSerialize::serialize(&pool, &mut serialized_data).unwrap();
            let fee: f64 = pool.max_fee_percentage.parse().unwrap();
            let liquidity: f64 = pool.liquidity.parse().unwrap();
            let item: PoolItem = PoolItem {
                mintA: pool.mint_x.clone(),
                mintB: pool.mint_y.clone(),
                vaultA: pool.reserve_x.clone(),
                vaultB: pool.reserve_y.clone(),
                tradeFeeRate: fee.clone() as u128,
            };
            pools_vec.push(item);

            let market: Market = Market {
                tokenMintA: pool.mint_x.clone(),
                tokenVaultA: pool.reserve_x.clone(),
                tokenMintB: pool.mint_y.clone(),
                tokenVaultB: pool.reserve_y.clone(),
                dexLabel: DexLabel::METEORA,
                fee: fee.clone() as u128,        
                id: pool.address.clone(),
                account_data: Some(serialized_data),
                liquidity: Some(liquidity as u128),
            };

            let pair_string = toPairString(pool.mint_x, pool.mint_y);
            if dex.pairToMarkets.contains_key(&pair_string.clone()) {
                let vec_market = dex.pairToMarkets.get_mut(&pair_string).unwrap();
                vec_market.push(market);
            } else {
                dex.pairToMarkets.insert(pair_string, vec![market]);
            }
        }

        info!("Meteora : {} pools founded", json_value.len());
        Self {
            dex: dex,
            pools: pools_vec,
        }
    }
}

pub async fn fetch_data_meteora() -> Result<(), Box<dyn std::error::Error>> {
    let response = get("https://dlmm-api.meteora.ag/pair/all").await?;
    // info!("response: {:?}", response);
    // info!("response-status: {:?}", response.status().is_success());
    if response.status().is_success() {
        let data = response.text().await?;
        
        match serde_json::from_str::<Root>(&data) {
            Ok(json) => {
                let file = File::create("src/markets/cache/meteora-markets.json")?;
                let mut writer = BufWriter::new(file);
                writer.write_all(serde_json::to_string(&json)?.as_bytes())?;
                writer.flush()?;
                info!("Data written to 'meteora-markets.json' successfully.");
            }
            Err(e) => {
                eprintln!("Failed to deserialize JSON: {:?}", e);
                // Optionally, save the raw JSON data to inspect it manually
                // let mut raw_file = File::create("src/markets/cache/meteora-markets-raw.json")?;
                // let mut writer = BufWriter::new(raw_file);
                // writer.write_all(data.as_bytes())?;
                // writer.flush()?;
                let result = print_json_segment("src/markets/cache/meteora-markets-raw.json", 3426919 - 100 as u64, 2000);
                // raw_file.write_all(data.as_bytes())?;
                // info!("Raw data written to 'meteora-markets-raw.json' for inspection.");
            }
        }
    } else {
        error!("Fetch of 'meteora-markets.json' not successful: {}", response.status());
    }
    Ok(())
}

pub async fn fetch_new_meteora_pools(rpc_client: &RpcClient, token: String, on_tokena: bool) -> Vec<(Pubkey, Market)> {

    let meteora_program = "LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo".to_string();
    // let pool = "5nRheYVXMTHEJXyAYG9KsUsXDTzvj9Las8M6NfNojaR".to_string();
    // println!("DEBUG ---- Token: {:?}", token);
    
    let mut new_markets: Vec<(Pubkey, Market)> = Vec::new(); 
    let filters = Some(vec![
        RpcFilterType::Memcmp(Memcmp::new(
            if on_tokena == true {
                88
            } else {
                120
            },
          MemcmpEncodedBytes::Base58(token.clone()),
        )),
        RpcFilterType::DataSize(904), 
    ]);
    
    let accounts = rpc_client.get_program_accounts_with_config(
        &from_str(&meteora_program).unwrap(),
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

    for account in accounts.clone() {
        println!("account data: {:?}", &account.1.data);
        let meteora_market = AccountData::try_from_slice(&account.1.data).unwrap();
        let market: Market = Market {
            tokenMintA: meteora_market.token_xmint.clone(),
            tokenVaultA: meteora_market.reserve_x.clone(),
            tokenMintB: meteora_market.token_ymint.clone(),
            tokenVaultB: meteora_market.reserve_y.clone(),
            dexLabel: DexLabel::METEORA,
            fee: 0 as u128,        
            id: from_Pubkey(account.0).clone(),
            account_data: Some(account.1.data),
            liquidity: Some(11111111111111 as u128),
        };
        new_markets.push((account.0, market));
    }
    println!("Accounts: {:?}", accounts);
    // println!("new_markets: {:?}", new_markets);
    return new_markets;
}

pub async fn stream_raydium(account: Pubkey) -> Result<()> {
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
                println!("account subscription error: {:?}", e);
                break;
            }
        }
    }

    Ok(())
}

// Simulate one route 
// I want to get the data of the market i'm interested in this route
pub async fn simulate_route_raydium(amount_in: f64, route: Route, market: Market, tokens_infos: HashMap<String, TokenInfos>) -> Result<(String, String), Box<dyn std::error::Error>> {
    // println!("account_data: {:?}", &market.account_data.clone().unwrap());
    // println!("market: {:?}", market.clone());
    let raydium_data = MeteoraPool::try_from_slice(&market.account_data.unwrap()).unwrap();
    // println!("raydium_data: {:?}", raydium_data);
    let decimals_0 = tokens_infos.get(&market.tokenMintA).unwrap().decimals;
    let decimals_1 = tokens_infos.get(&market.tokenMintB).unwrap().decimals;
    let mut params: String = String::new();

    let amount_in_uint = amount_in as u64;
    if route.token_0to1 {
        params = format!(
            "poolKeys={}&amountIn={}&currencyIn={}&decimalsIn={}&currencyOut={}&decimalsOut={}",
            market.id,
            amount_in_uint,
            market.tokenMintA,
            decimals_0,
            market.tokenMintB,
            decimals_1
        );
    } else {
        params = format!(
            "poolKeys={}&amountIn={}&currencyIn={}&decimalsIn={}&currencyOut={}&decimalsOut={}",
            market.id,
            amount_in_uint,
            market.tokenMintB,
            decimals_1,
            market.tokenMintA,
            decimals_0
        );
    }
    // Simulate a swap
    let env = Env::new();
    let domain = env.simulator_url;

    let req_url = format!("{}raydium_quote?{}", domain, params);
    // println!("req_url: {:?}", req_url);
    //URL like: http://localhost:3000/raydium_quote?poolKeys=58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2&amountIn=1000000&currencyIn=So11111111111111111111111111111111111111112&decimalsIn=9&currencyOut=EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v&decimalsOut=6
    
    let res = make_request(req_url).await?;
    let json_value: SimulationRes = res.json().await?;

    println!("estimatedAmountIn: {:?}", json_value.amountIn);
    println!("estimatedAmountOut: {:?}", json_value.estimatedAmountOut);
    println!("estimatedMinAmountOut: {:?}", json_value.estimatedMinAmountOut.clone().unwrap());
    
    Ok((
        json_value.estimatedAmountOut,
        json_value.estimatedMinAmountOut.unwrap_or_default(),
    ))

}

fn de_rating<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    Ok(match Value::deserialize(deserializer)? {
        Value::String(s) => s.parse().map_err(de::Error::custom)?,
        Value::Number(num) => num.as_f64().ok_or(de::Error::custom("Invalid number"))? as f64,
        Value::Null => 0.0,
        _ => return Err(de::Error::custom("wrong type"))
    })
}

pub type Root = Vec<MeteoraPool>;

#[derive(Default, BorshDeserialize, BorshSerialize, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MeteoraPool2 {
    pub address: String,
    #[serde(deserialize_with = "de_rating")]
    pub apr: f64,
    #[serde(deserialize_with = "de_rating")]
    pub apy: f64,
    pub base_fee_percentage: String,
    #[serde(deserialize_with = "de_rating")]
    pub bin_step: f64,
    pub cumulative_fee_volume: String,
    pub cumulative_trade_volume: String,
    #[serde(deserialize_with = "de_rating")]
    pub current_price: f64,
    #[serde(deserialize_with = "de_rating")]
    pub farm_apr: f64,
    #[serde(deserialize_with = "de_rating")]
    pub farm_apy: f64,
    #[serde(deserialize_with = "de_rating")]
    pub fees_24h: f64,
    pub hide: bool,
    pub liquidity: String,
    pub max_fee_percentage: String,
    pub mint_x: String,
    pub mint_y: String,
    pub name: String,
    pub protocol_fee_percentage: String,
    pub reserve_x: String,
    #[serde(deserialize_with = "de_rating")]
    pub reserve_x_amount: f64,
    pub reserve_y: String,
    #[serde(deserialize_with = "de_rating")]
    pub reserve_y_amount: f64,
    pub reward_mint_x: String,
    pub reward_mint_y: String,
    #[serde(deserialize_with = "de_rating")]
    pub today_fees: f64,
    #[serde(deserialize_with = "de_rating")]
    pub trade_volume_24h: f64,
}

#[derive(Default, BorshDeserialize, BorshSerialize, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MeteoraPool {
    pub address: String,
    pub name: String,
    #[serde(rename = "mint_x")]
    pub mint_x: String,
    #[serde(rename = "mint_y")]
    pub mint_y: String,
    #[serde(rename = "reserve_x")]
    pub reserve_x: String,
    #[serde(rename = "reserve_y")]
    pub reserve_y: String,
    #[serde(rename = "reserve_x_amount")]
    pub reserve_x_amount: i128,
    #[serde(rename = "reserve_y_amount")]
    pub reserve_y_amount: i128,
    #[serde(rename = "bin_step")]
    pub bin_step: i64,
    #[serde(rename = "base_fee_percentage")]
    pub base_fee_percentage: String,
    #[serde(rename = "max_fee_percentage")]
    pub max_fee_percentage: String,
    #[serde(rename = "protocol_fee_percentage")]
    pub protocol_fee_percentage: String,
    pub liquidity: String,
    #[serde(rename = "reward_mint_x")]
    pub reward_mint_x: String,
    #[serde(rename = "reward_mint_y")]
    pub reward_mint_y: String,
    #[serde(deserialize_with = "de_rating", rename = "fees_24h")]
    pub fees_24h: f64,
    #[serde(deserialize_with = "de_rating", rename = "today_fees")]
    pub today_fees: f64,
    #[serde(deserialize_with = "de_rating", rename = "trade_volume_24h")]
    pub trade_volume_24h: f64,
    #[serde(rename = "cumulative_trade_volume")]
    pub cumulative_trade_volume: String,
    #[serde(rename = "cumulative_fee_volume")]
    pub cumulative_fee_volume: String,
    #[serde(deserialize_with = "de_rating", rename = "current_price")]
    pub current_price: f64,
    #[serde(deserialize_with = "de_rating")]
    pub apr: f64,
    #[serde(deserialize_with = "de_rating")]
    pub apy: f64,
    #[serde(deserialize_with = "de_rating", rename = "farm_apr")]
    pub farm_apr: f64,
    #[serde(deserialize_with = "de_rating", rename = "farm_apy")]
    pub farm_apy: f64,
    pub hide: bool,
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////         ACCOUNT DATA            ///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



#[derive(Default, BorshDeserialize, BorshSerialize, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AccountData {
    pub parameters: Parameters,
    pub v_parameters: VParameters,
    pub bump_seed: Vec<i64>,
    pub bin_step_seed: Vec<i64>,
    pub pair_type: i64,
    pub active_id: i64,
    pub bin_step: i64,
    pub status: i64,
    pub padding1: Vec<i64>,
    #[serde(rename = "tokenXMint")]
    pub token_xmint: String,
    #[serde(rename = "tokenYMint")]
    pub token_ymint: String,
    pub reserve_x: String,
    pub reserve_y: String,
    pub protocol_fee: ProtocolFee,
    pub fee_owner: String,
    pub reward_infos: Vec<RewardInfo>,
    pub oracle: String,
    pub bin_array_bitmap: Vec<String>,
    pub last_updated_at: String,
    pub whitelisted_wallet: Vec<String>,
    pub base_key: String,
    pub activation_slot: String,
    pub swap_cap_deactivate_slot: String,
    pub max_swapped_amount: String,
    pub lock_durations_in_slot: String,
    pub creator: String,
    pub reserved: Vec<i64>,
}


#[derive(Default, BorshDeserialize, BorshSerialize, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Parameters {
    pub base_factor: i64,
    pub filter_period: i64,
    pub decay_period: i64,
    pub reduction_factor: i64,
    pub variable_fee_control: i64,
    pub max_volatility_accumulator: i64,
    pub min_bin_id: i64,
    pub max_bin_id: i64,
    pub protocol_share: i64,
    pub padding: Vec<i64>,
}


#[derive(Default, BorshDeserialize, BorshSerialize, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VParameters {
    pub volatility_accumulator: i64,
    pub volatility_reference: i64,
    pub index_reference: i64,
    pub padding: Vec<i64>,
    pub last_update_timestamp: String,
    pub padding1: Vec<i64>,
}


#[derive(Default, BorshDeserialize, BorshSerialize, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProtocolFee {
    pub amount_x: String,
    pub amount_y: String,
}


#[derive(Default, BorshDeserialize, BorshSerialize, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RewardInfo {
    pub mint: String,
    pub vault: String,
    pub funder: String,
    pub reward_duration: String,
    pub reward_duration_end: String,
    pub reward_rate: String,
    pub last_update_time: String,
    pub cumulative_seconds_with_empty_liquidity_reward: String,
}