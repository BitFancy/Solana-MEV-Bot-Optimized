pub fn toPairString(mintA: String, mintB: String) -> String {
    if (mintA < mintB) {
      return format!("{}/{}", mintA, mintB);
    } else {
        return format!("{}/{}", mintB, mintA);
    }
}