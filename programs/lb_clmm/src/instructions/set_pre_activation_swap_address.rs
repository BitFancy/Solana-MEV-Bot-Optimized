use crate::{errors::LBError, SetPreActivationInfo};
use anchor_lang::prelude::*;

pub fn handle(
    ctx: Context<SetPreActivationInfo>,
    pre_activation_swap_address: Pubkey,
) -> Result<()> {
    Ok(())
}
