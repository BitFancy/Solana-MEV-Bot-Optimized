use crate::arbitrage::types::{Route, TokenInfos};
use crate::markets::types::{Dex, DexLabel, Market, PoolItem, SimulationRes};
use crate::markets::utils::toPairString;
use crate::common::debug::print_json_segment;
use crate::common::utils::{from_Pubkey, from_str, make_request};
use crate::common::constants::Env;

use borsh::{BorshDeserialize, BorshSerialize};
use solana_client::rpc_filter::{Memcmp, MemcmpEncodedBytes, RpcFilterType};
use std::collections::HashMap;
use std::fs::File;
use std::fs;
use serde::{Deserialize, Deserializer, de, Serialize};
use serde_json::Value;
use reqwest::get;
use std::io::{BufWriter, Write};
use log::{info, error};
use solana_account_decoder::{UiAccountData, UiAccountEncoding};
use solana_program::pubkey::Pubkey;
use solana_sdk::commitment_config::CommitmentConfig;
use solana_pubsub_client::pubsub_client::PubsubClient;
use anyhow::Result;
use solana_client::rpc_client::RpcClient;
use solana_client::rpc_config::{RpcAccountInfoConfig, RpcProgramAccountsConfig};

use super::types::SimulationError;

#[derive(Debug)]
pub struct RaydiumDEX {
    pub dex: Dex,
    pub pools: Vec<PoolItem>,
}
impl RaydiumDEX {
    pub fn new(mut dex: Dex) -> Self {

        let mut pools_vec = Vec::new();
        
        let data = fs::read_to_string("src\\markets\\cache\\raydium-markets.json").expect("Error reading file");
        let json_value: Root  = serde_json::from_str(&data).unwrap();

        
        for pool in json_value.clone() {
            //Serialization foraccount_data
            let mut serialized_person: Vec<u8> = Vec::new();
            let result = BorshSerialize::serialize(&pool, &mut serialized_person).unwrap();
            let item: PoolItem = PoolItem {
                mintA: pool.base_mint.clone(),
                mintB: pool.quote_mint.clone(),
                vaultA: pool.base_mint.clone(),
                vaultB: pool.quote_mint.clone(),
                tradeFeeRate: pool.volume7d.clone() as u128,
            };
            pools_vec.push(item);

            let market: Market = Market {
                tokenMintA: pool.base_mint.clone(),
                tokenVaultA: pool.base_mint.clone(),
                tokenMintB: pool.quote_mint.clone(),
                tokenVaultB: pool.quote_mint.clone(),
                dexLabel: DexLabel::RAYDIUM,
                fee: pool.volume7d.clone() as u64,         //Not accurate, change this
                id: pool.amm_id.clone(),
                account_data: Some(serialized_person),
                liquidity: Some(pool.liquidity as u64),
            };

            let pair_string = toPairString(pool.base_mint, pool.quote_mint);
            if dex.pairToMarkets.contains_key(&pair_string.clone()) {
                let vec_market = dex.pairToMarkets.get_mut(&pair_string).unwrap();
                vec_market.push(market);
            } else {
                dex.pairToMarkets.insert(pair_string, vec![market]);
            }
        }

        info!("Raydium : {} pools founded", json_value.len());
        Self {
            dex: dex,
            pools: pools_vec,
        }
    }
  }

// pub async fn fetch_data_raydium() -> Result<(), Box<dyn std::error::Error>> {
//     let response = get("https://api.raydium.io/v2/main/pairs").await?;
//     // info!("response: {:?}", response);
//     // info!("response-status: {:?}", response.status().is_success());
//     if response.status().is_success() {
//         let json: Root = serde_json::from_str(&response.text().await?)?;        
//         // let json = &response.text().await?;        
//         info!("json: {:?}", json);
//         let mut file = File::create("src\\markets\\cache\\raydium-markets.json")?;
//         file.write_all(serde_json::to_string(&json)?.as_bytes())?;
//         info!("Data written to 'raydium-markets.json' successfully.");
//     } else {
//         info!("Fetch of 'raydium-markets.json' not successful: {}", response.status());
//     }
//     Ok(())
// }
pub async fn fetch_data_raydium() -> Result<(), Box<dyn std::error::Error>> {
    let response = get("https://api.raydium.io/v2/main/pairs").await?;
    // info!("response: {:?}", response);
    // info!("response-status: {:?}", response.status().is_success());
    if response.status().is_success() {
        let data = response.text().await?;
        
        match serde_json::from_str::<Root>(&data) {
            Ok(json) => {
                let file = File::create("src/markets/cache/raydium-markets.json")?;
                let mut writer = BufWriter::new(file);
                writer.write_all(serde_json::to_string(&json)?.as_bytes())?;
                writer.flush()?;
                info!("Data written to 'raydium-markets.json' successfully.");
            }
            Err(e) => {
                eprintln!("Failed to deserialize JSON: {:?}", e);
                // Optionally, save the raw JSON data to inspect it manually
                // let mut raw_file = File::create("src/markets/cache/raydium-markets-raw.json")?;
                let result = print_json_segment("src/markets/cache/raydium-markets.json", 21174733 - 1000 as u64, 2000);
                // raw_file.write_all(data.as_bytes())?;
                // info!("Raw data written to 'raydium-markets-raw.json' for inspection.");
            }
        }
    } else {
        error!("Fetch of 'raydium-markets.json' not successful: {}", response.status());
    }
    Ok(())
}

// pub async fn fetch_new_raydium_pools(rpc_client: &RpcClient, token: String, on_tokena: bool ) -> Vec<(Pubkey, Market)> {
pub async fn fetch_new_raydium_pools(rpc_client: &RpcClient, token: String, on_tokena: bool) -> Vec<(Pubkey, Market)> {

    let raydium_program = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8".to_string();
    
    let mut new_markets: Vec<(Pubkey, Market)> = Vec::new(); 
    let filters = Some(vec![
        RpcFilterType::Memcmp(Memcmp::new(
            if on_tokena == true {
                400
            } else {
                432
            },
          MemcmpEncodedBytes::Base58(token.clone()),
        )),
        RpcFilterType::DataSize(752), 
    ]);
    
    let accounts = rpc_client.get_program_accounts_with_config(
        &from_str(&raydium_program).unwrap(),
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
        let raydium_account = AmmInfo::try_from_slice(&account.1.data).unwrap();
        let fees: u128 = (raydium_account.fees.trade_fee_numerator / raydium_account.fees.trade_fee_denominator) as u128;
        let market: Market = Market {
            tokenMintA: from_Pubkey(raydium_account.coin_vault_mint.clone()),
            tokenVaultA: from_Pubkey(raydium_account.coin_vault.clone()),
            tokenMintB: from_Pubkey(raydium_account.pc_vault_mint.clone()),
            tokenVaultB: from_Pubkey(raydium_account.pc_vault.clone()),
            fee: fees as u64,
            dexLabel: DexLabel::RAYDIUM,
            id: from_Pubkey(account.0.clone()),
            account_data: Some(account.1.data),
            liquidity: Some(666 as u64),
        };
        new_markets.push((account.0, market));
    }
    // println!("Accounts: {:?}", accounts);
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
pub async fn simulate_route_raydium(printing_amt: bool, amount_in: u64, route: Route, market: Market, tokens_infos: HashMap<String, TokenInfos>) -> Result<(String, String), Box<dyn std::error::Error>> {
    // println!("account_data: {:?}", &market.account_data.clone().unwrap());
    // println!("market: {:?}", market.clone());
    // let raydium_data = AmmInfo::try_from_slice(&market.account_data.unwrap()).unwrap();
    // println!("raydium_data: {:?}", raydium_data);
    let token0 = tokens_infos.get(&market.tokenMintA).unwrap();
    let token1 = tokens_infos.get(&market.tokenMintB).unwrap();
    let mut params: String = String::new();

    let amount_in_uint = amount_in as u64;
    if route.token_0to1 {
        params = format!(
            "poolKeys={}&amountIn={}&currencyIn={}&decimalsIn={}&symbolTokenIn={}&currencyOut={}&decimalsOut={}&symbolTokenOut={}",
            market.id,
            amount_in_uint,
            market.tokenMintA,
            token0.decimals,
            token0.symbol,
            market.tokenMintB,
            token1.decimals,
            token1.symbol
        );
    } else {
        params = format!(
            "poolKeys={}&amountIn={}&currencyIn={}&decimalsIn={}&symbolTokenIn={}&currencyOut={}&decimalsOut={}&symbolTokenOut={}",
            market.id,
            amount_in_uint,
            market.tokenMintB,
            token1.decimals,
            token1.symbol,
            market.tokenMintA,
            token0.decimals,
            token0.symbol
        );
    }
    // Simulate a swap
    let env = Env::new();
    let domain = env.simulator_url;

    let req_url = format!("{}raydium_quote?{}", domain, params);
    
    let res = make_request(req_url).await?;
    let res_text = res.text().await?;

    if let Ok(json_value) = serde_json::from_str::<SimulationRes>(&res_text) {
        if printing_amt {
            println!("estimatedAmountIn: {:?} {:?}", json_value.amountIn, if route.token_0to1 == true { token0.clone().symbol } else { token1.clone().symbol });
            println!("estimatedAmountOut: {:?} {:?}", json_value.estimatedAmountOut, if route.token_0to1 == true { token1.clone().symbol } else { token0.clone().symbol });
            println!("estimatedMinAmountOut: {:?} {:?}", json_value.estimatedMinAmountOut.clone().unwrap(), if route.token_0to1 == true { token1.clone().symbol } else { token0.clone().symbol });
        }
        Ok((json_value.estimatedAmountOut,json_value.estimatedMinAmountOut.unwrap_or_default()))
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

fn de_rating<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    Ok(match Value::deserialize(deserializer)? {
        Value::String(s) => s.parse().map_err(de::Error::custom)?,
        Value::Number(num) => num.as_f64().ok_or(de::Error::custom("Invalid number"))? as f64,
        Value::Null => 0.0,
        _ => return Err(de::Error::custom("wrong type"))
    })
}

pub type Root = Vec<RaydiumPool>;

#[derive(Default, BorshDeserialize, BorshSerialize, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RaydiumPool {
    pub name: String,
    pub amm_id: String,
    pub lp_mint: String,
    pub base_mint: String,
    pub quote_mint: String,
    pub market: String,
    #[serde(deserialize_with = "de_rating")]
    pub liquidity: f64,
    #[serde(deserialize_with = "de_rating")]
    pub volume24h: f64,
    #[serde(deserialize_with = "de_rating")]
    pub volume24h_quote: f64,
    #[serde(deserialize_with = "de_rating")]
    pub fee24h: f64,
    #[serde(deserialize_with = "de_rating")]
    pub fee24h_quote: f64,
    #[serde(deserialize_with = "de_rating")]
    pub volume7d: f64,
    #[serde(deserialize_with = "de_rating")]
    pub volume7d_quote: f64,
    #[serde(deserialize_with = "de_rating")]
    pub fee7d: f64,
    #[serde(deserialize_with = "de_rating")]
    pub fee7d_quote: f64,
    #[serde(deserialize_with = "de_rating")]
    pub volume30d: f64,
    #[serde(deserialize_with = "de_rating")]
    pub volume30d_quote: f64,
    #[serde(deserialize_with = "de_rating")]
    pub fee30d: f64,
    #[serde(deserialize_with = "de_rating")]
    pub fee30d_quote: f64,
    #[serde(deserialize_with = "de_rating")]
    pub price: f64,
    #[serde(deserialize_with = "de_rating")]
    pub lp_price: f64,
    #[serde(deserialize_with = "de_rating")]
    pub token_amount_coin: f64,
    #[serde(deserialize_with = "de_rating")]
    pub token_amount_pc: f64,
    #[serde(deserialize_with = "de_rating")]
    pub token_amount_lp: f64,
    #[serde(deserialize_with = "de_rating")]
    pub apr24h: f64,
    #[serde(deserialize_with = "de_rating")]
    pub apr7d: f64,
    #[serde(deserialize_with = "de_rating")]
    pub apr30d: f64,
}

#[derive(Default, BorshDeserialize, BorshSerialize, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MarketStateLayoutV3 {
    pub func_signature: [u8; 5],
    pub account_flags: [u8; 8],
    pub owner_address: Pubkey,
    pub vault_signer_nonce: u64,
    pub base_mint: Pubkey,
    pub quote_mint: Pubkey,
    pub base_vault: Pubkey,
    pub base_deposits_total: u64,
    pub base_fees_accrued: u64,
    pub quote_vault: Pubkey,
    pub quote_deposits_total: u64,
    pub quote_fees_accrued: u64,
    pub quote_dust_threshold: u64,
    pub request_queue: Pubkey,
    pub event_queue: Pubkey,
    pub bids: Pubkey,
    pub asks: Pubkey,
    pub base_lot_size: u64,
    pub quote_lot_size: u64,
    pub fee_rate_bps: u64,
    pub referrer_rebates_accrued: u64,
    pub nope: [u8; 7],
}

//// Struct for Account Data on get_multiples_account
#[cfg_attr(feature = "client", derive(Debug))]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, BorshDeserialize, BorshSerialize)]
pub struct AmmInfo {
    /// Initialized status.
    pub status: u64,
    /// Nonce used in program address.
    /// The program address is created deterministically with the nonce,
    /// amm program id, and amm account pubkey.  This program address has
    /// authority over the amm's token coin account, token pc account, and pool
    /// token mint.
    pub nonce: u64,
    /// max order count
    pub order_num: u64,
    /// within this range, 5 => 5% range
    pub depth: u64,
    /// coin decimal
    pub coin_decimals: u64,
    /// pc decimal
    pub pc_decimals: u64,
    /// amm machine state
    pub state: u64,
    /// amm reset_flag
    pub reset_flag: u64,
    /// min size 1->0.000001
    pub min_size: u64,
    /// vol_max_cut_ratio numerator, sys_decimal_value as denominator
    pub vol_max_cut_ratio: u64,
    /// amount wave numerator, sys_decimal_value as denominator
    pub amount_wave: u64,
    /// coinLotSize 1 -> 0.000001
    pub coin_lot_size: u64,
    /// pcLotSize 1 -> 0.000001
    pub pc_lot_size: u64,
    /// min_cur_price: (2 * amm.order_num * amm.pc_lot_size) * max_price_multiplier
    pub min_price_multiplier: u64,
    /// max_cur_price: (2 * amm.order_num * amm.pc_lot_size) * max_price_multiplier
    pub max_price_multiplier: u64,
    /// system decimal value, used to normalize the value of coin and pc amount
    pub sys_decimal_value: u64,
    /// All fee information
    pub fees: Fees,
    /// Statistical data
    pub state_data: StateData,
    /// Coin vault
    pub coin_vault: Pubkey,
    /// Pc vault
    pub pc_vault: Pubkey,
    /// Coin vault mint
    pub coin_vault_mint: Pubkey,
    /// Pc vault mint
    pub pc_vault_mint: Pubkey,
    /// lp mint
    pub lp_mint: Pubkey,
    /// open_orders key
    pub open_orders: Pubkey,
    /// market key
    pub market: Pubkey,
    /// market program key
    pub market_program: Pubkey,
    /// target_orders key
    pub target_orders: Pubkey,
    /// padding
    pub padding1: [u64; 8],
    /// amm owner key
    pub amm_owner: Pubkey,
    /// pool lp amount
    pub lp_amount: u64,
    /// client order id
    pub client_order_id: u64,
    /// padding
    pub padding2: [u64; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, BorshDeserialize, BorshSerialize)]
pub struct Fees {
    /// numerator of the min_separate
    pub min_separate_numerator: u64,
    /// denominator of the min_separate
    pub min_separate_denominator: u64,

    /// numerator of the fee
    pub trade_fee_numerator: u64,
    /// denominator of the fee
    /// and 'trade_fee_denominator' must be equal to 'min_separate_denominator'
    pub trade_fee_denominator: u64,

    /// numerator of the pnl
    pub pnl_numerator: u64,
    /// denominator of the pnl
    pub pnl_denominator: u64,

    /// numerator of the swap_fee
    pub swap_fee_numerator: u64,
    /// denominator of the swap_fee
    pub swap_fee_denominator: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, BorshDeserialize, BorshSerialize)]
pub struct StateData {
    /// delay to take pnl coin
    pub need_take_pnl_coin: u64,
    /// delay to take pnl pc
    pub need_take_pnl_pc: u64,
    /// total pnl pc
    pub total_pnl_pc: u64,
    /// total pnl coin
    pub total_pnl_coin: u64,
    /// ido pool open time
    pub pool_open_time: u64,
    /// padding for future updates
    pub padding: [u64; 2],
    /// switch from orderbookonly to init
    pub orderbook_to_init_time: u64,

    /// swap coin in amount
    pub swap_coin_in_amount: u128,
    /// swap pc out amount
    pub swap_pc_out_amount: u128,
    /// charge pc as swap fee while swap pc to coin
    pub swap_acc_pc_fee: u64,

    /// swap pc in amount
    pub swap_pc_in_amount: u128,
    /// swap coin out amount
    pub swap_coin_out_amount: u128,
    /// charge coin as swap fee while swap coin to pc
    pub swap_acc_coin_fee: u64,
}