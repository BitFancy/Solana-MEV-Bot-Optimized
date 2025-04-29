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
use num::Integer;
use solana_client::rpc_client::RpcClient;
use solana_program::hash;
use solana_program::instruction::AccountMeta;
use std::rc::Rc;
use std::result::Result::Ok;
use std::mem::size_of;

use log::{info, error};
use solana_sdk::instruction::Instruction;
use solana_sdk::signature::read_keypair_file;
use spl_associated_token_account::instruction::create_associated_token_account;

use crate::common::constants::Env;
use crate::common::utils::from_str;
use crate::markets::meteora::AccountData;
use crate::markets::types::DexLabel;
use crate::transactions::create_transaction::{InstructionDetails, MarketInfos};


#[derive(Debug, Clone)]
pub struct SwapParametersMeteora {
    pub lb_pair: Pubkey,
    pub amount_in: u64,
    pub swap_for_y: bool,
    pub input_token: Pubkey,
    pub output_token: Pubkey,
    pub minimum_amount_out: u64,
}
// 
pub async fn construct_meteora_instructions(params: SwapParametersMeteora) -> Vec<InstructionDetails> {
    let SwapParametersMeteora {
        amount_in,
        lb_pair,
        swap_for_y,
        input_token,
        output_token,
        minimum_amount_out,
    } = params;
    // info!("METEORA CRAFT SWAP INSTRUCTION !");

    let mut swap_instructions: Vec<InstructionDetails> = Vec::new();
    let env = Env::new();
    let payer = read_keypair_file(env.payer_keypair_path).expect("Wallet keypair file not found");
    
    let amm_program = from_str("LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo").unwrap();
    
    let rpc_client: RpcClient = RpcClient::new(env.rpc_url);
    let pool_account: solana_sdk::account::Account = rpc_client.get_account(&lb_pair).unwrap();
    let pool_state = AccountData::try_from_slice(&pool_account.data).unwrap();
    
    // println!("Pool State: {:#?}", pool_state);

    //Get event authority
    let (event_authority, _bump) = Pubkey::find_program_address(&[b"__event_authority"], &amm_program);

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

    //Get bin arrays
    let active_bin_array_idx = bin_id_to_bin_array_index(pool_state.active_id).unwrap();
    let (bin_array_0, _bump) = derive_bin_array_pda(lb_pair, active_bin_array_idx as i64, amm_program);

    let bin_array_bitmap_extension = derive_bin_array_bitmap_extension(lb_pair, amm_program);
    let bin_array_1 = derive_bin_array_pda(lb_pair, (active_bin_array_idx - 1) as i64, amm_program).0;
    let bin_array_2 = derive_bin_array_pda(lb_pair, (active_bin_array_idx - 2) as i64, amm_program).0;
    

    let accounts = vec![
        // LbPair
        AccountMeta::new(lb_pair, false),
        AccountMeta::new_readonly(amm_program, false),
        AccountMeta::new(pool_state.reserve_x, false),
        AccountMeta::new(pool_state.reserve_y, false),
        //pda in
        AccountMeta::new(pda_user_source, false),
        //pda out
        AccountMeta::new(pda_user_destination, false),
        AccountMeta::new_readonly(pool_state.token_xmint, false),
        AccountMeta::new_readonly(pool_state.token_ymint, false),
        AccountMeta::new(pool_state.oracle, false),
        AccountMeta::new(amm_program, false),
        //user
        AccountMeta::new_readonly(payer.pubkey(), true),
        //token program
        AccountMeta::new_readonly(spl_token::id(), false),
        AccountMeta::new_readonly(spl_token::id(), false),
        //Event authority
        AccountMeta::new(event_authority, false),
        AccountMeta::new_readonly(amm_program, false),
        // 3 more accounts...
        AccountMeta::new(bin_array_0, false),
        AccountMeta::new(bin_array_1, false),
        AccountMeta::new(bin_array_2, false),
        // AccountMeta::new(from_str("EGcDqYxK3Ke7STeKwgaDH8uLSXqMEmwFC2hMnvWk7PwW").unwrap(), false),
    ];

    //Data Instruction
    let mut sighash = [0u8; 8];
    sighash.copy_from_slice(&hash::hash("global:swap".as_bytes()).to_bytes()[..8]);
    let mut data = [sighash].concat();
    data.extend_from_slice(&amount_in.to_le_bytes());
    data.extend_from_slice(&minimum_amount_out.to_le_bytes());

    let instruction = Instruction{
        program_id: amm_program,
        accounts,
        data,
    };

    swap_instructions.push(InstructionDetails{
        instruction: instruction, 
        details: "Meteora Swap Instruction".to_string(),
        market: Some(MarketInfos{dex_label: DexLabel::METEORA, address: lb_pair })
    });
    return swap_instructions;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////   UTILS   ///////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////
#[derive(BorshSerialize, BorshDeserialize)]
pub struct SwapInstructionBaseIn {
    pub amount_in: u64,
    pub minimum_amount_out: u64,
}

pub const BIN_ARRAY_BITMAP_SEED: &[u8] = b"bitmap";
pub const BIN_ARRAY: &[u8] = b"bin_array";
pub const MAX_BIN_PER_ARRAY: usize = 70;

pub fn derive_bin_array_bitmap_extension(lb_pair: Pubkey, amm_program: Pubkey) -> (Pubkey, u8) {
    let result = Pubkey::find_program_address(&[BIN_ARRAY_BITMAP_SEED, lb_pair.as_ref()], &amm_program);
    return result;
}

pub fn derive_bin_array_pda(lb_pair: Pubkey, bin_array_index: i64, amm_program: Pubkey) -> (Pubkey, u8) {
    let result = Pubkey::find_program_address(
        &[BIN_ARRAY, lb_pair.as_ref(), &bin_array_index.to_le_bytes()],
        &amm_program,
    );
    return result
}

/// Get bin array index from bin id
pub fn bin_id_to_bin_array_index(bin_id: i32) -> Result<i32> {
    let (idx, rem) = bin_id.div_rem(&(MAX_BIN_PER_ARRAY as i32));

    if bin_id.is_negative() && rem != 0 {
        Ok(idx.checked_sub(1).unwrap())
    } else {
        Ok(idx)
    }
}
