/**
 * Famous stock trader & investor quotes.
 * market: 'us' | 'pk' | 'both'
 * category: 'patience' | 'risk' | 'analysis' | 'mindset' | 'volatility'
 */
export const TRADER_QUOTES = [
  // ── Warren Buffett ─────────────────────────────────────────────────────────
  {
    quote: "The stock market is a device for transferring money from the impatient to the patient.",
    author: "Warren Buffett",
    role: "Chairman, Berkshire Hathaway",
    market: "both",
    category: "patience",
  },
  {
    quote: "Be fearful when others are greedy, and greedy when others are fearful.",
    author: "Warren Buffett",
    role: "Chairman, Berkshire Hathaway",
    market: "both",
    category: "mindset",
  },
  {
    quote: "Price is what you pay. Value is what you get.",
    author: "Warren Buffett",
    role: "Chairman, Berkshire Hathaway",
    market: "both",
    category: "analysis",
  },
  {
    quote: "It's far better to buy a wonderful company at a fair price than a fair company at a wonderful price.",
    author: "Warren Buffett",
    role: "Chairman, Berkshire Hathaway",
    market: "both",
    category: "analysis",
  },
  // ── Charlie Munger ─────────────────────────────────────────────────────────
  {
    quote: "Invert, always invert. Turn a situation or problem upside down. Look at it backwards.",
    author: "Charlie Munger",
    role: "Vice Chairman, Berkshire Hathaway",
    market: "both",
    category: "mindset",
  },
  {
    quote: "All I want to know is where I'm going to die, so I'll never go there.",
    author: "Charlie Munger",
    role: "Vice Chairman, Berkshire Hathaway",
    market: "both",
    category: "risk",
  },
  // ── Peter Lynch ────────────────────────────────────────────────────────────
  {
    quote: "Know what you own, and know why you own it.",
    author: "Peter Lynch",
    role: "Former Manager, Magellan Fund",
    market: "both",
    category: "analysis",
  },
  {
    quote: "The real key to making money in stocks is not to get scared out of them.",
    author: "Peter Lynch",
    role: "Former Manager, Magellan Fund",
    market: "both",
    category: "mindset",
  },
  {
    quote: "In this business, if you're good, you're right six times out of ten. You're never going to be right nine times out of ten.",
    author: "Peter Lynch",
    role: "Former Manager, Magellan Fund",
    market: "both",
    category: "mindset",
  },
  // ── Ray Dalio ──────────────────────────────────────────────────────────────
  {
    quote: "He who lives by the crystal ball will eat shattered glass.",
    author: "Ray Dalio",
    role: "Founder, Bridgewater Associates",
    market: "both",
    category: "risk",
  },
  {
    quote: "Diversification is the holy grail of investing.",
    author: "Ray Dalio",
    role: "Founder, Bridgewater Associates",
    market: "both",
    category: "risk",
  },
  {
    quote: "Pain plus reflection equals progress.",
    author: "Ray Dalio",
    role: "Founder, Bridgewater Associates",
    market: "both",
    category: "mindset",
  },
  // ── George Soros ───────────────────────────────────────────────────────────
  {
    quote: "It's not whether you're right or wrong, but how much money you make when you're right and how much you lose when you're wrong.",
    author: "George Soros",
    role: "Founder, Soros Fund Management",
    market: "us",
    category: "risk",
  },
  {
    quote: "Markets are constantly in a state of uncertainty and flux, and money is made by discounting the obvious and betting on the unexpected.",
    author: "George Soros",
    role: "Founder, Soros Fund Management",
    market: "us",
    category: "mindset",
  },
  // ── Benjamin Graham ────────────────────────────────────────────────────────
  {
    quote: "The intelligent investor is a realist who sells to optimists and buys from pessimists.",
    author: "Benjamin Graham",
    role: "Father of Value Investing",
    market: "both",
    category: "analysis",
  },
  {
    quote: "In the short run, the market is a voting machine, but in the long run, it is a weighing machine.",
    author: "Benjamin Graham",
    role: "Father of Value Investing",
    market: "both",
    category: "patience",
  },
  // ── Jesse Livermore ────────────────────────────────────────────────────────
  {
    quote: "The market is never wrong — opinions often are.",
    author: "Jesse Livermore",
    role: "Legendary Trader",
    market: "both",
    category: "mindset",
  },
  {
    quote: "There is nothing new in Wall Street. There can't be because speculation is as old as the hills. Whatever happens in the stock market today has happened before and will happen again.",
    author: "Jesse Livermore",
    role: "Legendary Trader",
    market: "us",
    category: "volatility",
  },
  // ── John Templeton ─────────────────────────────────────────────────────────
  {
    quote: "The four most expensive words in the English language are: 'This time it's different.'",
    author: "Sir John Templeton",
    role: "Pioneer of Global Investing",
    market: "both",
    category: "mindset",
  },
  {
    quote: "Bull markets are born on pessimism, grown on skepticism, mature on optimism, and die on euphoria.",
    author: "Sir John Templeton",
    role: "Pioneer of Global Investing",
    market: "both",
    category: "volatility",
  },
  // ── Paul Tudor Jones ───────────────────────────────────────────────────────
  {
    quote: "The secret to being successful from a trading perspective is to have an indefatigable and an undying and unquenchable thirst for information and knowledge.",
    author: "Paul Tudor Jones",
    role: "Founder, Tudor Investment Corp",
    market: "us",
    category: "analysis",
  },
  {
    quote: "Losers average losers.",
    author: "Paul Tudor Jones",
    role: "Founder, Tudor Investment Corp",
    market: "both",
    category: "risk",
  },
  // ── Pakistan / Emerging Markets ────────────────────────────────────────────
  {
    quote: "In emerging markets, patience is not just a virtue — it's your edge. Volatility punishes the impatient and rewards those who stay.",
    author: "Arif Habib",
    role: "Chairman, Arif Habib Group (PSX)",
    market: "pk",
    category: "patience",
  },
  {
    quote: "The KSE has rewarded disciplined investors who look past the noise of rupee fluctuations and geopolitical headlines.",
    author: "Mian Mansha",
    role: "Chairman, Nishat Group (PSX)",
    market: "pk",
    category: "patience",
  },
  {
    quote: "Pakistan's market has one of the highest earnings yields in Asia. The risk premium is real — but so is the opportunity.",
    author: "Market Proverb",
    role: "Pakistan Stock Exchange",
    market: "pk",
    category: "analysis",
  },
  {
    quote: "On the PSX, follow fundamentals, not WhatsApp tips.",
    author: "Common Wisdom",
    role: "PSX Veteran Traders",
    market: "pk",
    category: "mindset",
  },
  // ── General / Both ─────────────────────────────────────────────────────────
  {
    quote: "The trend is your friend until the end when it bends.",
    author: "Ed Seykota",
    role: "Legendary Trend Trader",
    market: "both",
    category: "volatility",
  },
  {
    quote: "Risk comes from not knowing what you're doing.",
    author: "Warren Buffett",
    role: "Chairman, Berkshire Hathaway",
    market: "both",
    category: "risk",
  },
  {
    quote: "In investing, what is comfortable is rarely profitable.",
    author: "Robert Arnott",
    role: "Chairman, Research Affiliates",
    market: "both",
    category: "mindset",
  },
  {
    quote: "Wide diversification is only required when investors do not understand what they are doing.",
    author: "Warren Buffett",
    role: "Chairman, Berkshire Hathaway",
    market: "both",
    category: "analysis",
  },
]

/**
 * Pick a random quote, optionally biased toward a market.
 * @param {'us'|'pk'|null} preferMarket - ticker market hint
 * @returns {{ quote, author, role, category }}
 */
export function getRandomQuote(preferMarket = null) {
  let pool = TRADER_QUOTES
  if (preferMarket === 'pk') {
    // 60% chance to pick a Pakistan-tagged quote, 40% both/us
    const pkPool = TRADER_QUOTES.filter(q => q.market === 'pk' || q.market === 'both')
    pool = Math.random() < 0.6 ? pkPool : TRADER_QUOTES
  } else if (preferMarket === 'us') {
    pool = TRADER_QUOTES.filter(q => q.market === 'us' || q.market === 'both')
  }
  return pool[Math.floor(Math.random() * pool.length)]
}

/**
 * Detect market from ticker string.
 * @param {string} ticker
 * @returns {'pk'|'us'}
 */
export function detectMarket(ticker = '') {
  return ticker.toUpperCase().includes('.KA') ? 'pk' : 'us'
}
