// //Taken here: https://github.com/MeteoraAg/dlmm-sdk/blob/main/cli/src/instructions/swap.rs

use anchor_client::solana_client::rpc_config::RpcSendTransactionConfig;
use anchor_client::{Client, Cluster};
use anchor_client::{solana_sdk::pubkey::Pubkey, solana_sdk::signer::Signer};
use anchor_lang::error_code;
use anchor_spl::associated_token::get_associated_token_address;
use anchor_spl::token::spl_token;
use anyhow::*;
use anchor_lang::solana_program::msg;

use borsh::{BorshDeserialize, BorshSerialize};
use num::FromPrimitive;
use num_bigint::{BigInt, BigUint};
use rust_decimal::{Decimal, MathematicalOps};
use serde::{Deserialize, Serialize};
use solana_client::rpc_client::RpcClient;
use solana_program::hash;
use solana_program::instruction::AccountMeta;
use std::rc::Rc;
use std::result::Result::Ok;
use std::mem::size_of;
use std::str::FromStr;

use log::{error, info};
use solana_sdk::instruction::Instruction;
use solana_sdk::signature::read_keypair_file;
use spl_associated_token_account::instruction::create_associated_token_account;

use crate::common::constants::Env;
use crate::common::utils::{from_str, make_request};
use crate::markets::meteora::AccountData;
use crate::markets::orca_whirpools::WhirlpoolAccountState;
use crate::markets::types::DexLabel;
use crate::transactions::create_transaction::{InstructionDetails, MarketInfos};


#[derive(Debug, Clone)]
pub struct SwapParametersOrcaWhirpools {
    pub whirpools: Pubkey,
    pub amount_in: u64,
    pub input_token: Pubkey,
    pub output_token: Pubkey,
    pub minimum_amount_out: u64,
}
// 
pub async fn construct_orca_whirpools_instructions(params: SwapParametersOrcaWhirpools) -> Vec<InstructionDetails> {
    let SwapParametersOrcaWhirpools {
        whirpools,
        amount_in,
        input_token,
        output_token,
        minimum_amount_out,
    } = params;
    // info!("ORCA WHIRPOOLS CRAFT SWAP INSTRUCTION !");
    // println!("Params: {:?}", params);

    let mut swap_instructions: Vec<InstructionDetails> = Vec::new();
    let env = Env::new();
    let payer = read_keypair_file(env.payer_keypair_path).expect("Wallet keypair file not found");
    
    let amm_program = from_str("whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc").unwrap();
    
    let rpc_client: RpcClient = RpcClient::new(env.rpc_url);
    let pool_account: solana_sdk::account::Account = rpc_client.get_account(&whirpools).unwrap();
    // println!("Params: {:?}", pool_account);
    // println!("Params data length: {:?}", pool_account.data.len());

    let pool_state = WhirlpoolAccountState::try_from_slice(&pool_account.data).unwrap();

    // println!("Pool State: {:#?}", pool_state);
    
    if input_token != pool_state.token_mint_a && input_token != pool_state.token_mint_b {
        error!("TokenIn don't match with any token on the pool");
        return swap_instructions
    }
    if output_token != pool_state.token_mint_a && output_token != pool_state.token_mint_b {
        error!("TokenOut don't match with any token on the pool");
        return swap_instructions
    }
    
    let a_to_b: bool = if input_token == pool_state.token_mint_a { true } else { false };

    //Get PDA
    let pda_user_source = get_associated_token_address(
        &payer.pubkey(),
        &input_token,
    );
    match rpc_client.get_account(&pda_user_source) {
        Ok(account) => {}
        Err(error) => {
            // error!("❌ PDA not exist for {}", input_token);
        }
    }

    let pda_user_destination = get_associated_token_address(
        &payer.pubkey(),
        &output_token,
    );

    match rpc_client.get_account(&pda_user_destination) {
        Ok(account) => {}
        Err(error) => {
            // error!("❌ PDA not exist for {}", output_token);
        }
    }

    let params = format!(
        "tickCurrentIndex={}&tickSpacing={}&aToB={}&programId={}&whirlpoolAddress={}",
        pool_state.tick_current_index,
        pool_state.tick_spacing,
        a_to_b,
        amm_program,
        whirpools
    );

    let domain = env.simulator_url;

    let req_url = format!("{}whirpools_tick_arrays?{}", domain, params);
    // println!("req_url: {:?}", req_url);
    
    let res = make_request(req_url).await.expect("Error in request to simulator for tick arrays");
    let res_text = res.text().await.expect("Error in request to simulator for tick arrays");
    
    let tick_arrays = serde_json::from_str::<TickArraysRes>(&res_text).expect("Unwrap error in tick arrays");

    let accounts = vec![
        // TokenProgram
        AccountMeta::new_readonly(spl_token::id(), false),
        //Token Authority / User ? 
        AccountMeta::new(payer.pubkey(), true),
        AccountMeta::new(whirpools, false),
        AccountMeta::new(
            if a_to_b {
                pda_user_source
            } else {
                pda_user_destination
            }
            , false
        ),
        AccountMeta::new(pool_state.token_vault_a, false),
        AccountMeta::new(
            if a_to_b {
                pda_user_destination
            } else {
                pda_user_source
            }
            , false
        ),
        AccountMeta::new(pool_state.token_vault_b, false),
        //Tick arrays
        AccountMeta::new(from_str(tick_arrays.tick_array_0.as_str()).unwrap(), false),
        AccountMeta::new(from_str(tick_arrays.tick_array_1.as_str()).unwrap(), false),
        AccountMeta::new(from_str(tick_arrays.tick_array_2.as_str()).unwrap(), false),
        //Oracle
        AccountMeta::new_readonly(from_str(tick_arrays.oracle.as_str()).unwrap(), false),
    ];

    //Data Instruction
    let other_amount_threshold: u64 = minimum_amount_out;
    //SqrtPrice with 1% slippage
    let sqrt_price_limit_1percent = pool_state.sqrt_price / 100;
    let sqrt_price_limit = pool_state.sqrt_price - sqrt_price_limit_1percent;

    // println!("sqrt_price_limit {:?}", pool_state.sqrt_price);
    // println!("Computed sqrt_price_limit {:?}", sqrt_price_limit);

    let mut sighash = [0u8; 8];
    sighash.copy_from_slice(&hash::hash("global:swap".as_bytes()).to_bytes()[..8]);
    let mut data = [sighash].concat();
    data.extend_from_slice(&amount_in.to_le_bytes());
    data.extend_from_slice(&other_amount_threshold.to_le_bytes());
    data.extend_from_slice(&sqrt_price_limit.to_le_bytes());
    data.extend_from_slice(&[1]);   //True
    data.extend_from_slice(&[1]);   //True
        
    let instruction = Instruction{
        program_id: amm_program,
        accounts,
        data,
    };
    // println!("Instruction: {:?}", instruction);

    swap_instructions.push(InstructionDetails{
        instruction: instruction, 
        details: "Orca Whirpool Swap Instruction".to_string(),
        market: Some(MarketInfos{dex_label: DexLabel::ORCA_WHIRLPOOLS, address: whirpools })
    });

    return swap_instructions;

}





////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////   UTILS   ///////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////
#[derive(BorshSerialize, BorshDeserialize)]
pub struct SwapInstructionBaseIn {
    pub amount_in: u64,
    pub other_amount_threshold: u64,
    pub sqrt_price_limit: u128,
    pub amount_specified_is_input: bool,
    pub a_to_b: bool,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Serialize, Deserialize)]
pub struct TickArraysRes {
    pub tick_array_0: String,
    pub tick_array_1: String,
    pub tick_array_2: String,
    pub oracle: String,
}

// pub const MAX_SWAP_TICK_ARRAYS: i32 = 3;
// const TICK_ARRAY_SIZE: i32 = 10;
// const MIN_TICK_INDEX: i32 = -100000;
// const MAX_TICK_INDEX: i32 = 100000;
// const PDA_TICK_ARRAY_SEED: &[u8] = b"tick_array_seed";

// pub struct TickArrayRequest {
//     whirlpool_address: Pubkey,
//     a_to_b: bool,
//     tick_current_index: i32,
//     tick_spacing: u16,
// }

// pub async fn get_batch_tick_arrays(
//     program_id: Pubkey,
//     tick_array_requests: Vec<TickArrayRequest>,
//     opts: Option<WhirlpoolAccountFetchOptions>
// ) -> Vec<Vec<TickArray>> {
//     let mut addresses: Vec<Pubkey> = Vec::new();
//     let mut request_to_indices = Vec::new();

//     for request in tick_array_requests.iter() {
//         let TickArrayRequest { tick_current_index, tick_spacing, a_to_b, whirlpool_address } = request;
//         let request_addresses = get_tick_array_public_keys(
//             tick_current_index.clone(),
//             tick_spacing.clone(),
//             a_to_b.clone(),
//             program_id.clone(),
//             whirlpool_address.clone()
//         );
//         request_to_indices.push((addresses.len(), addresses.len() + request_addresses.len()));
//         addresses.extend(request_addresses);
//     }

//     let data = get_tick_arrays(addresses.clone(), opts).await;

//     request_to_indices.into_iter().map(|(start, end)| {
//         let address_slice = &addresses[start..end];
//         let data_slice = &data[start..end];
//         address_slice.iter().zip(data_slice.iter()).map(|(addr, data)| TickArray {
//             address: addr.clone(),
//             data: data.data.clone()
//         }).collect()
//     }).collect()
// }


// // https://github.com/orca-so/whirlpools/blob/3dc98d0/sdk/src/utils/public/swap-utils.ts#L93
// pub fn get_tick_array_public_keys(
//         tick_current_index: i32,
//         tick_spacing: u16,
//         a_to_b: bool,
//         program_id: Pubkey,
//         whirlpool_address: Pubkey
//     ) -> Vec<Pubkey> {

//     let shift = if a_to_b == true { 0 } else { tick_spacing };
//     let mut offset = 0;
//     let mut tick_array_addresses: Vec<Pubkey> = Vec::new();

//     for n in 0..MAX_SWAP_TICK_ARRAYS {
//         let start_index = get_start_tick_index(tick_current_index + shift as i32, tick_spacing as i32, offset);
//         if start_index == 0 {
//             //Error
//             let mut empty_vec: Vec<Pubkey> = Vec::new();
//             return empty_vec;
//         }

//         let pda = get_tick_arrays_fetcher(&program_id, &whirlpool_address, start_index);
//         tick_array_addresses.push(pda.0);
//         offset = if a_to_b == true { offset - 1 } else { offset + 1 };
//     }

//     return tick_array_addresses;
// }

// pub fn get_start_tick_index(tick_index: i32, tick_spacing: i32, offset: i32) -> i32 {
//     let real_index = (tick_index / tick_spacing / TICK_ARRAY_SIZE) as i32;
//     let start_tick_index = (real_index + offset) * tick_spacing * TICK_ARRAY_SIZE;

//     let ticks_in_array = TICK_ARRAY_SIZE * tick_spacing;
//     let min_tick_index = MIN_TICK_INDEX - ((MIN_TICK_INDEX % ticks_in_array) + ticks_in_array);
    
//     if start_tick_index < min_tick_index {
//         error!("startTickIndex is too small - {}", start_tick_index);
//         return 0    //Assumtion that 0 is not a good value
//     }
//     if start_tick_index > MAX_TICK_INDEX {
//         error!("startTickIndex is too large - {}", start_tick_index);
//         return 0    //Assumtion that 0 is not a good value
//     }

//     start_tick_index
// }

// pub fn get_tick_arrays_fetcher(program_id: &Pubkey, whirlpool_address: &Pubkey, start_tick: i32) -> (Pubkey, u8) {
//     let seed = PDA_TICK_ARRAY_SEED;
//     let whirlpool_buffer = whirlpool_address.to_bytes();
//     let start_tick_str = start_tick.to_string();
//     let start_tick_buffer = start_tick_str.as_bytes();

//     let seeds = [&seed[..], &whirlpool_buffer[..], &start_tick_buffer[..]];
    
//     Pubkey::find_program_address(&seeds, program_id)
// }

fn to_x64(num: u128) -> u128 {
    let result = num * 2_u128.pow(64);
    return result
}