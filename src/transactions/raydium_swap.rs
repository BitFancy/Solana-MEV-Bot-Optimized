//Taken here: https://github.com/MeteoraAg/dlmm-sdk/blob/main/cli/src/instructions/swap.rs

use anchor_client::solana_client::rpc_config::RpcSendTransactionConfig;
use anchor_client::{solana_sdk::pubkey::Pubkey, solana_sdk::signer::Signer};
use anchor_spl::associated_token::get_associated_token_address;

use anchor_spl::token::spl_token;
use borsh::BorshDeserialize;
// use anyhow::*;

use log::{info, error};
use raydium_amm::instruction::swap_base_in;
use solana_client::rpc_client::RpcClient;
use solana_sdk::instruction::Instruction;
use solana_sdk::signature::read_keypair_file;
use spl_associated_token_account::instruction::create_associated_token_account;

use crate::common::constants::Env;
use crate::common::utils::from_str;
use crate::markets::raydium::AmmInfo;
use crate::markets::types::DexLabel;
use crate::transactions::create_transaction::{InstructionDetails, MarketInfos};

use super::utils::get_keys_for_market;

#[derive(Debug, Clone)]
pub struct SwapParametersRaydium {
    pub pool: Pubkey,
    pub input_token_mint: Pubkey,
    pub output_token_mint: Pubkey,
    pub amount_in: u64,
    pub swap_for_y: bool,
    pub min_amount_out: u64
}
// Function are imported from Raydium library, you can see here: 
// https://github.com/raydium-io/raydium-library
pub fn construct_raydium_instructions(params: SwapParametersRaydium) -> Vec<InstructionDetails> {
    let SwapParametersRaydium {
        pool,
        input_token_mint,
        output_token_mint,
        amount_in,
        swap_for_y,
        min_amount_out
    } = params;
    // info!("RAYDIUM CRAFT SWAP INSTRUCTION !");

    let mut swap_instructions: Vec<InstructionDetails> = Vec::new();
    let env = Env::new();
    let payer = read_keypair_file(env.payer_keypair_path).expect("Wallet keypair file not found");
    
    let amm_program = from_str("675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8").unwrap();
    //Devnet : HWy1jotHpo6UqeQxx49dpYYdQB8wj9Qk9MdxwjLvDHB8
    //Mainnet : 675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8
    
    let rpc_client: RpcClient = RpcClient::new(env.rpc_url);
    let pool_account: solana_sdk::account::Account = rpc_client.get_account(&pool).unwrap();
    // println!("Params data length: {:?}", pool_account.data.len());
    let pool_state = AmmInfo::try_from_slice(&pool_account.data).unwrap();
    // println!("min_amount_out: {:?}", min_amount_out);
    // println!("Params: {:?}", params);
    // println!("Pool State: {:?}", pool_state);

    let authority = raydium_amm::processor::Processor::authority_id(
        &amm_program,
        raydium_amm::processor::AUTHORITY_AMM,
        pool_state.nonce as u8,
    ).unwrap();

    // load market keys
    let market_keys = get_keys_for_market(
        &rpc_client,
        &pool_state.market_program,
        &pool_state.market,
    ).unwrap();

    let pda_user_source = get_associated_token_address(
        &payer.pubkey(),
        &input_token_mint,
    );
    match rpc_client.get_account(&pda_user_source) {
        Ok(account) => {}
        Err(error) => {
            // error!("❌ PDA not exist for {}", input_token_mint);
        }
    }

    let pda_user_destination = get_associated_token_address(
        &payer.pubkey(),
        &output_token_mint,
    );

    match rpc_client.get_account(&pda_user_destination) {
        Ok(account) => {}
        Err(error) => {
            // error!("❌ PDA not exist for {}", output_token_mint);
        }
    }

    let amm_target_orders = from_str("9DCxsMizn3H1hprZ7xWe6LDzeUeZBksYFpBWBtSf1PQX").unwrap();

    let swap_instruction = swap_base_in(
        &amm_program,
        &pool,
        &authority,
        &pool_state.open_orders,
        &amm_target_orders,
        &pool_state.coin_vault,
        &pool_state.pc_vault,
        &pool_state.market_program,
        &pool_state.market,
        &market_keys.bids,
        &market_keys.asks,
        &market_keys.event_q,
        &market_keys.coin_vault,
        &market_keys.pc_vault,
        &market_keys.vault_signer_key,
        &pda_user_source,
        &pda_user_destination,
        &payer.pubkey(),
        amount_in,
        min_amount_out,
    ).expect("Error in Raydium swap instruction construction");

    // println!("DATA: {:?}", swap_instruction.data);
    swap_instructions.push(InstructionDetails{ 
        instruction: swap_instruction, 
        details: "Raydium Swap Instruction".to_string(), 
        market: Some(MarketInfos{dex_label: DexLabel::RAYDIUM, address: pool })
    });

    return (swap_instructions);
}
