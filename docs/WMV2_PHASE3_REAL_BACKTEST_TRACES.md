# WMv2 Phase 3 — Real Backtest Forecast + Leakage-Audit Traces
*Per-question forecasts for every arm plus the strict as-of evidence trace (retrieval window, document publication dates, per-document temporal status, and claim-level leakage partitions) used for the manual leakage audit. Source of truth: `experiments/results/phase3/real_backtest.json`.*

## Manual leakage audit — stratified sample
For each question below, the `as_of` is strictly before the resolution date. The retrieval layer pairs `after:`/`before:` on Google News RSS, runs per-document temporal verification, and a claim-level leakage audit (post-as-of publication, resolution-term language, retrospective phrasing) before freezing the bundle. The audit columns below are what a human checks: are any admitted documents published **after** `as_of`? Do any leakage flags fire? 

### `trump_2024` — Will Donald Trump win the 2024 US presidential election?
- domain **elections**, as_of **2024-10-20**, horizon **2024-11-06**, realized outcome **1** — Trump won; called 2024-11-06.
- status **completed_with_degradation**, support grade **exploratory**, latency 68.8s
- forecasts — prior_only **0.5000**, phase2_no_posterior **0.6000**, phase3_posterior **0.4667**, point_estimate **0.4000**, market **—**
- posterior: prior_mean 0.5000 → posterior_mean 0.3506 (shift -0.1494); included claims 9 → 8 effective observations; consumed True; reproducible_hash True
- **as-of audit**: bundle_hash `bcd64dbf208a17f6`, as_of 2024-10-20, 13 docs, included 9 / excluded 0 / suspicious 0 claims, leakage flags 0

  | doc | source | published | temporal_status |
  |---|---|---|---|
  | a40a0c6b21ed8d8a | Brookings | 2024-08-02 | likely_pre_asof |
  | a579213964a98f19 | Council on Foreign Relations | 2024-09-13 | likely_pre_asof |
  | 8afb2a05f8207a92 | ABC7 Bay Area | 2024-07-22 | likely_pre_asof |
  | 45ac91e358ad7086 | Northeastern Global News | 2024-07-23 | likely_pre_asof |
  | 5340823a32c8544f | NPR | 2024-07-31 | likely_pre_asof |
  | cd22ee3b3ac67f84 | WHYY | 2024-09-10 | likely_pre_asof |
  | 8dafdc13451b4832 | YouGov | 2024-09-10 | likely_pre_asof |
  | a5d7f1832ff6f728 | ABC7 New York | 2024-07-22 | likely_pre_asof |
  | a5e0215b986e9848 | CalMatters | 2024-07-21 | likely_pre_asof |
  | 6e28c6df155ffcf6 | PBS | 2024-09-10 | likely_pre_asof |
  | ca027fa247f10c25 | Pew Research Center | 2024-08-26 | likely_pre_asof |
  | 092f4e0380ccfe49 | ABC11 News | 2024-09-14 | likely_pre_asof |

### `harris_2024` — Will Kamala Harris win the 2024 US presidential election?
- domain **elections**, as_of **2024-10-20**, horizon **2024-11-06**, realized outcome **0** — Harris lost the 2024 election.
- status **completed_with_degradation**, support grade **exploratory**, latency 70.3s
- forecasts — prior_only **0.5000**, phase2_no_posterior **0.5500**, phase3_posterior **0.5000**, point_estimate **0.5500**, market **—**
- posterior: prior_mean 0.5000 → posterior_mean 0.5609 (shift 0.0609); included claims 12 → 8 effective observations; consumed True; reproducible_hash True
- **as-of audit**: bundle_hash `5f46d9be7cc474a9`, as_of 2024-10-20, 25 docs, included 12 / excluded 0 / suspicious 0 claims, leakage flags 0

  | doc | source | published | temporal_status |
  |---|---|---|---|
  | a40a0c6b21ed8d8a | Brookings | 2024-08-02 | likely_pre_asof |
  | 5340823a32c8544f | NPR | 2024-07-31 | likely_pre_asof |
  | a579213964a98f19 | Council on Foreign Relations | 2024-09-13 | likely_pre_asof |
  | cd22ee3b3ac67f84 | WHYY | 2024-09-10 | likely_pre_asof |
  | a5e0215b986e9848 | CalMatters | 2024-07-21 | likely_pre_asof |
  | 8dafdc13451b4832 | YouGov | 2024-09-10 | likely_pre_asof |
  | 3181d27644015ce5 | ABC7 Bay Area | 2024-09-01 | likely_pre_asof |
  | f26af18c0b5deb47 | Time Magazine | 2024-07-21 | likely_pre_asof |
  | 6e28c6df155ffcf6 | PBS | 2024-09-10 | likely_pre_asof |
  | 092f4e0380ccfe49 | ABC11 News | 2024-09-14 | likely_pre_asof |
  | 6d88a2c9c00030c5 | FairVote | 2024-07-15 | likely_pre_asof |
  | 550e7671e6ff9947 | Rutgers University | 2024-07-30 | likely_pre_asof |

### `biden_nominee` — Will Joe Biden be the Democratic nominee for the 2024 US presidential election?
- domain **elections**, as_of **2024-07-05**, horizon **2024-08-22**, realized outcome **0** — Biden withdrew 2024-07-21; Harris nominated.
- status **completed_with_degradation**, support grade **highly_speculative**, latency 62.4s
- forecasts — prior_only **0.5897**, phase2_no_posterior **0.6226**, phase3_posterior **0.7925**, point_estimate **0.7547**, market **—**
- posterior: prior_mean 0.5897 → posterior_mean 0.6602 (shift 0.0705); included claims 13 → 8 effective observations; consumed True; reproducible_hash True
- **as-of audit**: bundle_hash `2d09f2bac1614b04`, as_of 2024-07-05, 13 docs, included 13 / excluded 0 / suspicious 0 claims, leakage flags 0

  | doc | source | published | temporal_status |
  |---|---|---|---|
  | a94af34102292c29 | PBS | 2024-03-12 | likely_pre_asof |
  | 6f1b0a079015ecad | ABC News - Breaking News, Latest News and Videos | 2024-07-02 | likely_pre_asof |
  | 81692debc2cac37a | NBC News | 2024-06-28 | likely_pre_asof |
  | d3b2860dba618545 | Time Magazine | 2024-06-28 | likely_pre_asof |
  | dc9b925fb59634c1 | Los Angeles Times | 2024-06-28 | likely_pre_asof |
  | c8bc6386fedd2bf2 | Politico | 2024-03-12 | likely_pre_asof |
  | ea8d457b86fffd2d | AP News | 2024-07-01 | likely_pre_asof |
  | ba2d70adee41e3a2 | WTJX Newsfeed | 2024-06-09 | likely_pre_asof |
  | 8965f1c888ffddf7 | Wisconsin Examiner | 2024-03-19 | likely_pre_asof |
  | 827b55e75a8715d6 | Democratic Party of Wisconsin | 2024-06-09 | likely_pre_asof |
  | 84f9a14732c8d580 | Spectrum News | 2024-04-14 | likely_pre_asof |
  | e85b3470c6988941 | Democrats Abroad | 2024-04-11 | likely_pre_asof |

### `uk_labour` — Will the Labour Party win the 2024 United Kingdom general election?
- domain **elections**, as_of **2024-06-25**, horizon **2024-07-04**, realized outcome **1** — Labour won a majority 2024-07-04.
- status **completed_with_degradation**, support grade **highly_speculative**, latency 85.3s
- forecasts — prior_only **0.5000**, phase2_no_posterior **0.6438**, phase3_posterior **0.6027**, point_estimate **0.6712**, market **—**
- posterior: prior_mean 0.5000 → posterior_mean 0.5824 (shift 0.0824); included claims 11 → 8 effective observations; consumed True; reproducible_hash True
- **as-of audit**: bundle_hash `d6807c96fb829d13`, as_of 2024-06-25, 37 docs, included 11 / excluded 0 / suspicious 0 claims, leakage flags 0

  | doc | source | published | temporal_status |
  |---|---|---|---|
  | a440d8f4753f5677 | YouGov | 2024-06-19 | likely_pre_asof |
  | 9dff3835b84cadba | The Guardian | 2024-05-22 | likely_pre_asof |
  | 84ab32d43beb4fc6 | Reuters | 2024-06-20 | likely_pre_asof |
  | 32674163a22ae9cd | Al Jazeera | 2024-05-23 | likely_pre_asof |
  | 364277cd0306d822 | BBC | 2024-05-22 | likely_pre_asof |
  | 82943eb7dfa83497 | The Conversation | 2024-06-21 | likely_pre_asof |
  | c2c209198197f103 | Loughborough University | 2024-06-07 | likely_pre_asof |
  | c9428be1c257785b | CityNews Kitchener | 2024-06-07 | likely_pre_asof |
  | 029e77d4a00b65e2 | The Telegraph | 2024-06-01 | likely_pre_asof |
  | 4b534cb83337ff34 | The Independent | 2024-05-23 | likely_pre_asof |
  | 02c21778607c8c60 | Indy100 | 2024-06-14 | likely_pre_asof |
  | 7d677ffdd3829db6 | The Guardian | 2024-06-21 | likely_pre_asof |

### `shutdown_oct24` — Will there be a US federal government shutdown on October 1, 2024?
- domain **politics**, as_of **2024-09-20**, horizon **2024-10-01**, realized outcome **0** — CR signed 2024-09-26; no shutdown.
- status **completed_with_degradation**, support grade **highly_speculative**, latency 78.6s
- forecasts — prior_only **0.5000**, phase2_no_posterior **0.5152**, phase3_posterior **0.8182**, point_estimate **0.8030**, market **—**
- posterior: prior_mean 0.5000 → posterior_mean 0.7352 (shift 0.2352); included claims 17 → 8 effective observations; consumed True; reproducible_hash True
- **as-of audit**: bundle_hash `c439d7235714116d`, as_of 2024-09-20, 15 docs, included 17 / excluded 0 / suspicious 0 claims, leakage flags 0

  | doc | source | published | temporal_status |
  |---|---|---|---|
  | d23baff13e5a5dac | PBS | 2024-09-18 | likely_pre_asof |
  | 17af730c42ef5bc0 | NBC News | 2024-09-09 | likely_pre_asof |
  | 59b0435ae36bc7f1 | New Jersey Monitor | 2024-09-18 | likely_pre_asof |
  | 944075ba807d1b7e | Oklahoma Voice | 2024-09-10 | likely_pre_asof |
  | 915a373304d71a00 | Maine Morning Star | 2024-09-10 | likely_pre_asof |
  | b5206b106b53227e | Business Insider | 2024-09-11 | likely_pre_asof |
  | d3910a3f013b293a | FOX40 | 2024-09-18 | likely_pre_asof |
  | 96801f104f740474 | Arkansas Advocate | 2024-09-18 | likely_pre_asof |
  | 92a563b0d5d6a264 | The Indiana Lawyer | 2024-09-10 | likely_pre_asof |
  | f4c8a01f47d9ea19 | The Hill | 2024-09-13 | likely_pre_asof |
  | c8552baab1eec64a | Spectrum News | 2024-09-20 | uncertain |
  | 4b7793a3f6a9aa19 | Politico | 2024-09-17 | likely_pre_asof |

### `shutdown_dec24` — Will there be a US federal government shutdown before the end of 2024?
- domain **politics**, as_of **2024-12-10**, horizon **2024-12-31**, realized outcome **0** — CR passed 2024-12-21; no shutdown.
- status **completed_with_degradation**, support grade **highly_speculative**, latency 76.0s
- forecasts — prior_only **0.5000**, phase2_no_posterior **0.5949**, phase3_posterior **0.7595**, point_estimate **0.7595**, market **—**
- posterior: prior_mean 0.5000 → posterior_mean 0.6692 (shift 0.1692); included claims 10 → 8 effective observations; consumed True; reproducible_hash True
- **as-of audit**: bundle_hash `ca4ec73a16ba786a`, as_of 2024-12-10, 22 docs, included 10 / excluded 0 / suspicious 0 claims, leakage flags 0

  | doc | source | published | temporal_status |
  |---|---|---|---|
  | 801b595f3ebe1229 | PBS | 2024-09-11 | likely_pre_asof |
  | e76851a46334158a | ABC News - Breaking News, Latest News and Videos | 2024-09-06 | likely_pre_asof |
  | 9460d8887f2829c1 | Kentucky Lantern | 2024-09-10 | likely_pre_asof |
  | 59b0435ae36bc7f1 | New Jersey Monitor | 2024-09-18 | likely_pre_asof |
  | a6db2cbc50810ea0 | NBC News | 2024-09-18 | likely_pre_asof |
  | b6a02f3789e91e26 | CBS News | 2024-09-25 | likely_pre_asof |
  | f68ad9bc37a36eee | The Guardian | 2024-09-26 | likely_pre_asof |
  | d3910a3f013b293a | FOX40 | 2024-09-18 | likely_pre_asof |
  | 0568f37a83ca32eb | Arab News | 2024-09-19 | likely_pre_asof |
  | 35f420c4dead744a | Democracy Docket | 2024-09-25 | likely_pre_asof |
  | 1cb0ee046d622798 | PBS | 2024-09-25 | likely_pre_asof |
  | 45874c8256da47fb | Cleveland.com | 2024-09-18 | likely_pre_asof |

### `fed_sep24` — Will the US Federal Reserve cut interest rates at its September 2024 meeting?
- domain **econ**, as_of **2024-09-10**, horizon **2024-09-19**, realized outcome **1** — FOMC cut 50bp 2024-09-18.
- status **completed_with_degradation**, support grade **exploratory**, latency 65.9s
- forecasts — prior_only **0.5000**, phase2_no_posterior **0.4000**, phase3_posterior **0.4000**, point_estimate **0.4000**, market **—**
- posterior: prior_mean 0.5000 → posterior_mean 0.6673 (shift 0.1673); included claims 15 → 8 effective observations; consumed True; reproducible_hash True
- **as-of audit**: bundle_hash `623d21888eeb256c`, as_of 2024-09-10, 18 docs, included 15 / excluded 0 / suspicious 1 claims, leakage flags 0

  | doc | source | published | temporal_status |
  |---|---|---|---|
  | fefa12d3fea3f984 | WSJ | 2024-07-18 | likely_pre_asof |
  | 34cd116770dda54a | Reuters | 2024-09-06 | likely_pre_asof |
  | b7f605bc1d8f052a | AgWeb | 2024-07-08 | likely_pre_asof |
  | 86b1efd7fd75e977 | Fortune | 2024-07-02 | likely_pre_asof |
  | be3d5e2e4c374570 | Business Insider | 2024-08-07 | likely_pre_asof |
  | 4346cb412f1eff57 | Finimize | 2024-09-06 | likely_pre_asof |
  | 75f919fb1bb08058 | Reuters | 2024-07-19 | likely_pre_asof |
  | dcf9a9337a0c0edb | CNBC | 2024-09-06 | likely_pre_asof |
  | 1a407fe54df4afb6 | WSJ | 2024-09-06 | likely_pre_asof |
  | 0cecf29025efe0c1 | Fortune | 2024-07-26 | likely_pre_asof |
  | 87b6e23371ee7eb5 | BusinessWorld Online | 2024-07-18 | likely_pre_asof |
  | efa60562ccd937d6 | Reuters | 2024-07-15 | likely_pre_asof |

### `fed_nov24` — Will the US Federal Reserve cut interest rates at its November 2024 meeting?
- domain **econ**, as_of **2024-10-28**, horizon **2024-11-08**, realized outcome **1** — FOMC cut 25bp 2024-11-07.
- status **completed_with_degradation**, support grade **exploratory**, latency 67.8s
- forecasts — prior_only **0.5000**, phase2_no_posterior **0.4200**, phase3_posterior **0.4200**, point_estimate **0.4200**, market **—**
- posterior: prior_mean 0.5000 → posterior_mean 0.7496 (shift 0.2496); included claims 16 → 8 effective observations; consumed True; reproducible_hash True
- **as-of audit**: bundle_hash `e0e218f39699edd5`, as_of 2024-10-28, 24 docs, included 16 / excluded 0 / suspicious 0 claims, leakage flags 0

  | doc | source | published | temporal_status |
  |---|---|---|---|
  | fefa12d3fea3f984 | WSJ | 2024-07-18 | likely_pre_asof |
  | b7f605bc1d8f052a | AgWeb | 2024-07-08 | likely_pre_asof |
  | 4ae79ef004bf51f6 | The New York Times | 2024-09-18 | likely_pre_asof |
  | 3182571cf2452783 | Moomoo | 2024-09-18 | likely_pre_asof |
  | 583001181a219e8f | Kansas Reflector | 2024-08-03 | likely_pre_asof |
  | 0cecf29025efe0c1 | Fortune | 2024-07-26 | likely_pre_asof |
  | d170783422fb9cf2 | New York Magazine | 2024-09-18 | likely_pre_asof |
  | 811b6fd2d2341a85 | tovima.com | 2024-09-18 | likely_pre_asof |
  | 1a407fe54df4afb6 | WSJ | 2024-09-06 | likely_pre_asof |
  | 1482cbc8b360bfc1 | Marianas Variety | 2024-07-18 | likely_pre_asof |
  | ec1d1d687095480b | Wikipedia | 2024-09-24 | likely_pre_asof |
  | 5dc88c0582de6976 | Brookings | 2024-08-01 | likely_pre_asof |

### `fed_dec24` — Will the US Federal Reserve cut interest rates at its December 2024 meeting?
- domain **econ**, as_of **2024-12-05**, horizon **2024-12-19**, realized outcome **1** — FOMC cut 25bp 2024-12-18.
- status **completed_with_degradation**, support grade **highly_speculative**, latency 89.1s
- forecasts — prior_only **0.5000**, phase2_no_posterior **0.4242**, phase3_posterior **0.4242**, point_estimate **0.4242**, market **—**
- posterior: prior_mean 0.5000 → posterior_mean 0.7297 (shift 0.2297); included claims 11 → 8 effective observations; consumed True; reproducible_hash True
- **as-of audit**: bundle_hash `6612452a14af70fb`, as_of 2024-12-05, 24 docs, included 11 / excluded 0 / suspicious 0 claims, leakage flags 0

  | doc | source | published | temporal_status |
  |---|---|---|---|
  | a47cb08a1e47048a | PBS | 2024-12-04 | uncertain |
  | 6d4880a257608b62 | CBS News | 2024-09-18 | likely_pre_asof |
  | 12da3c4bb86aa2e2 | The New York Times | 2024-08-23 | likely_pre_asof |
  | 451f4822bffbd5af | Reuters | 2024-09-11 | likely_pre_asof |
  | 4fa97b9294774426 | NPR | 2024-08-23 | likely_pre_asof |
  | 016baa06f950c65e | The Washington Post | 2024-09-18 | likely_pre_asof |
  | e2bd48d68ae0abab | NBC News | 2024-08-23 | likely_pre_asof |
  | 51a56aa4754c445c | Investopedia | 2024-09-18 | likely_pre_asof |
  | 852936cc704517cf | CNBC | 2024-11-07 | likely_pre_asof |
  | 0078a7e16fceb305 | CBS News | 2024-11-07 | likely_pre_asof |
  | 7e88af819f7f0688 | Reuters | 2024-10-29 | likely_pre_asof |
  | 542ddc833a647073 | The Washington Post | 2024-08-22 | likely_pre_asof |

### `fed_jan25` — Will the US Federal Reserve cut interest rates at its January 2025 meeting?
- domain **econ**, as_of **2025-01-20**, horizon **2025-01-30**, realized outcome **0** — FOMC held steady 2025-01-29.
- status **completed_with_degradation**, support grade **highly_speculative**, latency 74.2s
- forecasts — prior_only **0.5000**, phase2_no_posterior **0.4225**, phase3_posterior **0.3099**, point_estimate **0.3380**, market **—**
- posterior: prior_mean 0.5000 → posterior_mean 0.3574 (shift -0.1426); included claims 12 → 8 effective observations; consumed True; reproducible_hash True
- **as-of audit**: bundle_hash `1b70c16ba8de41ea`, as_of 2025-01-20, 12 docs, included 12 / excluded 0 / suspicious 0 claims, leakage flags 0

  | doc | source | published | temporal_status |
  |---|---|---|---|
  | ffd70ae111a2612e | CNBC | 2025-01-08 | likely_pre_asof |
  | b687f66e24636910 | WSJ | 2024-12-18 | likely_pre_asof |
  | d1a779be51430bc1 | Morningstar | 2024-12-19 | likely_pre_asof |
  | fc4421a4f52d1e1f | PBS | 2024-12-18 | likely_pre_asof |
  | 3b74f105c4372547 | Investopedia | 2024-12-18 | likely_pre_asof |
  | fa5afc071d37b23c | Reuters | 2024-12-18 | likely_pre_asof |
  | 4736efd13fdb1359 | ING THINK economic and financial analysis | ING THINK | 2024-12-18 | likely_pre_asof |
  | f649ecee958d9c2d | Federal Reserve Bank of San Francisco | 2025-01-05 | likely_pre_asof |
  | 9006692f3229aed5 | The New York Times | 2024-12-18 | likely_pre_asof |
  | 6e38ee55884abc93 | AIER | 2024-09-28 | likely_pre_asof |
  | 7955459013fbf631 | Washington State Standard | 2024-11-07 | likely_pre_asof |
  | aca21c987bc28068 | Business Insider | 2024-12-18 | likely_pre_asof |

### `btc_100k` — Will Bitcoin exceed one hundred thousand US dollars by the end of 2024?
- domain **finance**, as_of **2024-11-15**, horizon **2024-12-31**, realized outcome **1** — BTC passed $100k 2024-12-04.
- status **completed_with_degradation**, support grade **exploratory**, latency 95.8s
- forecasts — prior_only **0.5000**, phase2_no_posterior **0.6000**, phase3_posterior **0.5333**, point_estimate **0.6333**, market **—**
- posterior: prior_mean 0.5000 → posterior_mean 0.5725 (shift 0.0725); included claims 12 → 8 effective observations; consumed True; reproducible_hash True
- **as-of audit**: bundle_hash `4c95a69dfcf30ed7`, as_of 2024-11-15, 18 docs, included 12 / excluded 0 / suspicious 0 claims, leakage flags 0

  | doc | source | published | temporal_status |
  |---|---|---|---|
  | fea4ae1c5805757f | The New Yorker | 2024-08-05 | likely_pre_asof |
  | 509ff69b1bcecba5 | The New Yorker | 2024-10-07 | likely_pre_asof |
  | 7f4929e10f0918de | akingump.com | 2024-10-07 | likely_pre_asof |
  | 6952d48828ff3860 | Fortune | 2024-08-06 | likely_pre_asof |
  | 9be340be6e49bb86 | Bybit Learn | 2024-10-01 | likely_pre_asof |
  | 7e393f9d4ffd35f3 | TradingView | 2024-10-12 | likely_pre_asof |
  | f376cbe1f8c503ed | Federal Reserve (.gov) | 2024-07-22 | likely_pre_asof |
  | 429eec7124df82d4 | Peterson Institute for International Economics | 2024-08-26 | likely_pre_asof |
  | 6f9e68a6ccd8c5e1 | Consumer Financial Services Law Monitor | 2024-08-27 | likely_pre_asof |
  | 333699cf55cd4cc3 | Chicago - Federal Reserve Bank | 2024-11-14 | uncertain |
  | 8cd3fdc03c8ee04a | Bank Policy Institute | 2024-10-16 | likely_pre_asof |
  | 070c497b5df33d6e | Skadden, Arps, Slate, Meagher & Flom LLP | 2024-09-24 | likely_pre_asof |

### `recession_24` — Will the United States enter a recession in 2024?
- domain **macro**, as_of **2024-07-01**, horizon **2024-12-31**, realized outcome **0** — No NBER recession in 2024.
- status **completed_with_degradation**, support grade **highly_speculative**, latency 58.9s
- forecasts — prior_only **0.5000**, phase2_no_posterior **0.5063**, phase3_posterior **0.5949**, point_estimate **0.5443**, market **—**
- posterior: prior_mean 0.5000 → posterior_mean 0.5256 (shift 0.0256); included claims 8 → 3 effective observations; consumed True; reproducible_hash True
- **as-of audit**: bundle_hash `2a37fdb92be79a04`, as_of 2024-07-01, 3 docs, included 8 / excluded 0 / suspicious 1 claims, leakage flags 0

  | doc | source | published | temporal_status |
  |---|---|---|---|
  | 45636bf7f10ce06e | European Central Bank | 2024-05-28 | likely_pre_asof |
  | 00bc8c88eb57dfd6 | Substack | 2024-05-28 | likely_pre_asof |
  | dc1a21275a018d55 | Wikipedia | 2024-06-17 | likely_pre_asof |

### `nvda_split` — Will Nvidia announce a stock split in 2024?
- domain **finance**, as_of **2024-05-01**, horizon **2024-06-30**, realized outcome **1** — Nvidia announced 10-for-1 split 2024-05-22.
- status **completed_with_degradation**, support grade **highly_speculative**, latency 67.4s
- forecasts — prior_only **0.5000**, phase2_no_posterior **0.5606**, phase3_posterior **0.5000**, point_estimate **0.5152**, market **—**
- posterior: prior_mean 0.5000 → posterior_mean 0.5054 (shift 0.0054); included claims 8 → 5 effective observations; consumed True; reproducible_hash True
- **as-of audit**: bundle_hash `b5ddea9f76366849`, as_of 2024-05-01, 6 docs, included 8 / excluded 0 / suspicious 1 claims, leakage flags 0

  | doc | source | published | temporal_status |
  |---|---|---|---|
  | b12aa750188f5fdf | Forbes | 2024-02-22 | likely_pre_asof |
  | c1dc73e59fd7d9d1 | Fox Business | 2024-03-19 | likely_pre_asof |
  | 91687fa63c9e8db4 | CNBC | 2024-03-21 | likely_pre_asof |
  | e3ca1870878589ad | Barron's | 2024-03-20 | likely_pre_asof |
  | bd2a05812f0b45bc | Investopedia | 2024-03-20 | likely_pre_asof |
  | 712cacdcfa12ae0d | The Motley Fool | 2024-05-01 | uncertain |

### `sp500_6000` — Will the S&P 500 index close above 6000 in 2024?
- domain **finance**, as_of **2024-11-01**, horizon **2024-12-31**, realized outcome **1** — S&P 500 first closed >6000 2024-11-08.
- status **completed_with_degradation**, support grade **highly_speculative**, latency 84.0s
- forecasts — prior_only **0.5000**, phase2_no_posterior **0.5758**, phase3_posterior **0.5758**, point_estimate **0.6364**, market **—**
- posterior: prior_mean 0.5000 → posterior_mean 0.5566 (shift 0.0566); included claims 10 → 8 effective observations; consumed True; reproducible_hash True
- **as-of audit**: bundle_hash `2ffd0b09c3481777`, as_of 2024-11-01, 29 docs, included 10 / excluded 0 / suspicious 0 claims, leakage flags 0

  | doc | source | published | temporal_status |
  |---|---|---|---|
  | 136c93f734d26277 | CME Group | 2024-08-13 | likely_pre_asof |
  | 011a0365704254e2 | Barron's | 2024-10-04 | likely_pre_asof |
  | 57eaa08d7fc3c343 | KITCO | 2024-10-07 | likely_pre_asof |
  | 28c7febc662a4af4 | USFunds | 2024-09-27 | likely_pre_asof |
  | 74cf1335286bff61 | Times of Malta | 2024-10-16 | likely_pre_asof |
  | ecaf61557bd81e26 | FactSet Insight | 2024-08-02 | likely_pre_asof |
  | 89896302d62c12a8 | Neuberger Berman | 2024-07-31 | likely_pre_asof |
  | 2fa7dacced32a4c6 | Investopedia | 2024-07-10 | likely_pre_asof |
  | 109f7f3f76c3f6f7 | Business Insider | 2024-10-07 | likely_pre_asof |
  | 55ca4bff39e65ede | Reuters | 2024-10-11 | likely_pre_asof |
  | 56c7a3cd018233a5 | Bloomberg.com | 2024-08-10 | likely_pre_asof |
  | 3aff026c29313304 | FactSet Insight | 2024-11-01 | uncertain |

### `gpt5_2024` — Will OpenAI release a model called GPT-5 in 2024?
- domain **tech**, as_of **2024-08-01**, horizon **2024-12-31**, realized outcome **0** — No GPT-5 in 2024.
- status **completed_with_degradation**, support grade **highly_speculative**, latency 71.7s
- forecasts — prior_only **0.5000**, phase2_no_posterior **0.4507**, phase3_posterior **0.5915**, point_estimate **0.6197**, market **—**
- posterior: prior_mean 0.5000 → posterior_mean 0.6396 (shift 0.1396); included claims 11 → 8 effective observations; consumed True; reproducible_hash True
- **as-of audit**: bundle_hash `04d4e030da048007`, as_of 2024-08-01, 19 docs, included 11 / excluded 0 / suspicious 0 claims, leakage flags 0

  | doc | source | published | temporal_status |
  |---|---|---|---|
  | 5a304dabced558ff | CNBC | 2024-07-18 | likely_pre_asof |
  | 3d9a5cf76607e2e3 | Business Insider | 2024-05-22 | likely_pre_asof |
  | b3ec78273955e2ef | The New York Times | 2024-05-28 | likely_pre_asof |
  | 63b31014c587a161 | TechCrunch | 2024-08-01 | uncertain |
  | 56f8343398fafb65 | Reuters | 2024-05-14 | likely_pre_asof |
  | a035c639f39aaec8 | Fortune | 2024-05-03 | likely_pre_asof |
  | d4461321b6395d4d | The Washington Post | 2024-05-28 | likely_pre_asof |
  | 361ef63a8e6a4b5b | Time Magazine | 2024-06-07 | likely_pre_asof |
  | e8a1e48ed13a6904 | VentureBeat | 2024-05-21 | likely_pre_asof |
  | 451d399c2358425c | GeekWire | 2024-05-21 | likely_pre_asof |
  | 9df32c8870cce318 | CNBC | 2024-07-31 | uncertain |
  | d6350c6867cae9e2 | Reuters | 2024-04-12 | likely_pre_asof |

### `gpt5_2025` — Will OpenAI release a model called GPT-5 in 2025?
- domain **tech**, as_of **2025-06-01**, horizon **2025-12-31**, realized outcome **1** — OpenAI released GPT-5 2025-08.
- status **completed_with_degradation**, support grade **highly_speculative**, latency 91.8s
- forecasts — prior_only **0.5000**, phase2_no_posterior **0.3924**, phase3_posterior **0.2532**, point_estimate **0.2152**, market **—**
- posterior: prior_mean 0.5000 → posterior_mean 0.3973 (shift -0.1027); included claims 18 → 8 effective observations; consumed True; reproducible_hash True
- **as-of audit**: bundle_hash `ecf536fee050eb52`, as_of 2025-06-01, 37 docs, included 18 / excluded 0 / suspicious 0 claims, leakage flags 0

  | doc | source | published | temporal_status |
  |---|---|---|---|
  | 8bb3d5e3a9c17f5d | Tech Policy Press | 2025-05-08 | likely_pre_asof |
  | b533f140d93c880b | CNBC | 2025-02-27 | likely_pre_asof |
  | 088932c13b8ebd32 | PYMNTS.com | 2025-04-04 | likely_pre_asof |
  | e3157829e88bde80 | Fortune | 2025-02-25 | likely_pre_asof |
  | 4f9b711badcbefc6 | TechCrunch | 2025-03-05 | likely_pre_asof |
  | c5246ba70b3d7803 | The American Prospect | 2025-03-25 | likely_pre_asof |
  | 29bf4fc1a775e8a1 | GeekWire | 2025-05-08 | likely_pre_asof |
  | d51b293e0b7f1da7 | Business Insider | 2025-05-15 | likely_pre_asof |
  | cc38f3962ceb9904 | TechTarget | 2025-04-17 | likely_pre_asof |
  | 2f76678df060686b | fanaticalfuturist.com | 2025-05-27 | likely_pre_asof |
  | ab3e4c0903a0f268 | Bloomberg.com | 2025-02-02 | likely_pre_asof |
  | 5ba13bc23d7ba5ec | Al Jazeera | 2025-02-14 | likely_pre_asof |

### `apple_intel` — Will Apple release its Apple Intelligence features in 2024?
- domain **tech**, as_of **2024-08-01**, horizon **2024-12-31**, realized outcome **1** — Apple Intelligence launched 2024-10.
- status **completed_with_degradation**, support grade **exploratory**, latency 89.1s
- forecasts — prior_only **0.5897**, phase2_no_posterior **0.6000**, phase3_posterior **0.7000**, point_estimate **0.7333**, market **—**
- posterior: prior_mean 0.5897 → posterior_mean 0.6849 (shift 0.0952); included claims 12 → 7 effective observations; consumed True; reproducible_hash True
- **as-of audit**: bundle_hash `d7d03e3075be5714`, as_of 2024-08-01, 25 docs, included 12 / excluded 0 / suspicious 0 claims, leakage flags 0

  | doc | source | published | temporal_status |
  |---|---|---|---|
  | 4735e1c78f4a81f0 | Al Jazeera | 2024-06-11 | likely_pre_asof |
  | 0830f42210e4dc65 | CNBC | 2024-06-11 | likely_pre_asof |
  | 63a27289a71e9a1c | The Guardian | 2024-06-10 | likely_pre_asof |
  | bccc3011797740eb | observer.com | 2024-06-21 | likely_pre_asof |
  | dcb93ed8fd4cca7e | TechCrunch | 2024-06-10 | likely_pre_asof |
  | 0334892fd6102719 | EL PAÍS English | 2024-06-14 | likely_pre_asof |
  | e24d6fdb34f70054 | The Washington Post | 2024-06-10 | likely_pre_asof |
  | 6c3b448e5525f7b8 | Investopedia | 2024-06-10 | likely_pre_asof |
  | 5917b3559978fd8d | The Indian Express | 2024-06-11 | likely_pre_asof |
  | b74e0ad7efbb631e | Wikipedia | 2024-07-30 | likely_pre_asof |
  | 26aa59f7c4e68bd9 | Digital Markets Act | 2024-06-24 | likely_pre_asof |
  | 7b8af381af15bd68 | TechCrunch | 2024-07-24 | likely_pre_asof |

### `gaza_ceasefire24` — Will Israel and Hamas agree to a ceasefire by the end of 2024?
- domain **geopolitics**, as_of **2024-10-01**, horizon **2024-12-31**, realized outcome **0** — No ceasefire in 2024; deal reached Jan 2025.
- status **completed_with_degradation**, support grade **highly_speculative**, latency 83.4s
- forecasts — prior_only **0.5000**, phase2_no_posterior **0.3625**, phase3_posterior **0.2125**, point_estimate **0.1625**, market **—**
- posterior: prior_mean 0.5000 → posterior_mean 0.3934 (shift -0.1066); included claims 12 → 8 effective observations; consumed True; reproducible_hash True
- **as-of audit**: bundle_hash `a82c98850726176b`, as_of 2024-10-01, 13 docs, included 12 / excluded 0 / suspicious 0 claims, leakage flags 0

  | doc | source | published | temporal_status |
  |---|---|---|---|
  | 81673ff9c085ac53 | PBS | 2024-08-21 | likely_pre_asof |
  | 0f1745156cffa1b4 | CNN | 2024-08-16 | likely_pre_asof |
  | 1d30deecd7ae00b8 | Al Jazeera | 2024-08-24 | likely_pre_asof |
  | 4c2ab324b537ada6 | BBC | 2024-06-10 | likely_pre_asof |
  | 322e3a20fe94a189 | NPR | 2024-08-12 | likely_pre_asof |
  | 8fae47b2772c2d28 | AP News | 2024-08-14 | likely_pre_asof |
  | e7ff2d27d81f8213 | Le Monde.fr | 2024-06-07 | likely_pre_asof |
  | e6d6ddaffd88e375 | Reuters | 2024-06-05 | likely_pre_asof |
  | 579581680dff76eb | The New York Times | 2024-09-04 | likely_pre_asof |
  | 9f01c1ea93434ecb | BBC | 2024-07-04 | likely_pre_asof |
  | 28a976138ab87428 | Reuters | 2024-08-15 | likely_pre_asof |
  | 30b542ec8a5f68d1 | BBC | 2024-08-19 | likely_pre_asof |

### `gaza_ceasefire25` — Will an Israel-Hamas ceasefire take effect in January 2025?
- domain **geopolitics**, as_of **2025-01-10**, horizon **2025-01-31**, realized outcome **1** — Ceasefire took effect 2025-01-19.
- status **completed_with_degradation**, support grade **highly_speculative**, latency 90.9s
- forecasts — prior_only **0.5000**, phase2_no_posterior **0.3750**, phase3_posterior **0.4625**, point_estimate **0.4875**, market **—**
- posterior: prior_mean 0.5000 → posterior_mean 0.5742 (shift 0.0742); included claims 13 → 7 effective observations; consumed True; reproducible_hash True
- **as-of audit**: bundle_hash `81cecc873462aa77`, as_of 2025-01-10, 33 docs, included 13 / excluded 0 / suspicious 0 claims, leakage flags 0

  | doc | source | published | temporal_status |
  |---|---|---|---|
  | 5146c4745a35c669 | EL PAÍS English | 2025-01-09 | uncertain |
  | 0005386e942f23fc | Atlantic Council | 2024-12-19 | likely_pre_asof |
  | eb754719afd010b8 | BBC | 2024-11-09 | likely_pre_asof |
  | 78a7734b92fdfa3c | Al Jazeera Centre for Studies | 2024-12-11 | likely_pre_asof |
  | d618ecd572d9c5ed | ایران اینترنشنال | 2025-01-02 | likely_pre_asof |
  | 0e168d983a7191f5 | Washington Blade | 2024-10-22 | likely_pre_asof |
  | 439e799e073a951a | Atlantic Council | 2024-10-04 | likely_pre_asof |
  | 2fbad289ce769154 | The Washington Post | 2024-11-21 | likely_pre_asof |
  | 73a3307f037d2681 | Wikipedia | 2024-12-31 | likely_pre_asof |
  | 84b1a0d99899e52b | Quincy Institute for Responsible Statecraft | 2024-11-22 | likely_pre_asof |
  | 52be2b94df47ee9e | Jewish Telegraphic Agency | 2024-11-27 | likely_pre_asof |
  | 82b504d89aa1832c | The Times of Israel | 2024-11-01 | likely_pre_asof |

### `assad_fall` — Will Bashar al-Assad's government fall in Syria in 2024?
- domain **geopolitics**, as_of **2024-11-25**, horizon **2024-12-31**, realized outcome **1** — Assad government fell 2024-12-08.
- status **completed_with_degradation**, support grade **highly_speculative**, latency 77.8s
- forecasts — prior_only **0.4103**, phase2_no_posterior **0.3235**, phase3_posterior **0.1176**, point_estimate **0.1324**, market **—**
- posterior: prior_mean 0.4103 → posterior_mean 0.3269 (shift -0.0834); included claims 11 → 8 effective observations; consumed True; reproducible_hash True
- **as-of audit**: bundle_hash `e32144ffc645ebf6`, as_of 2024-11-25, 25 docs, included 11 / excluded 0 / suspicious 0 claims, leakage flags 0

  | doc | source | published | temporal_status |
  |---|---|---|---|
  | 601011ded9b55bed | Royal United Services Institute (RUSI) | 2024-08-06 | likely_pre_asof |
  | 95bfa1b358a332f5 | Al Jazeera | 2024-08-25 | likely_pre_asof |
  | a4911740c03f9285 | Brookings | 2024-08-26 | likely_pre_asof |
  | c2c76400b8f8c7c8 | arabnews.jp | 2024-10-17 | likely_pre_asof |
  | 03cdb874a53c9d1a | Atlantic Council | 2024-08-16 | likely_pre_asof |
  | 4735a54ad3f6fe59 | Reuters | 2024-09-25 | likely_pre_asof |
  | c55a362745fa7c7a | Carnegie Endowment for International Peace | 2024-09-19 | likely_pre_asof |
  | e2ea1791078ef301 | IRIS - Institut de relations internationales et stratégiques | 2024-11-20 | likely_pre_asof |
  | a64b85a0923433f9 | The Washington Institute | 2024-08-16 | likely_pre_asof |
  | f5f70d081210cb57 | The New Yorker | 2024-11-04 | likely_pre_asof |
  | c0f070511474df20 | Asia News Network | 2024-11-06 | likely_pre_asof |
  | b3172929bd99f996 | Enab Baladi | 2024-09-29 | likely_pre_asof |

### `ru_ua_cf24` — Will Russia and Ukraine agree to a ceasefire in 2024?
- domain **geopolitics**, as_of **2024-06-01**, horizon **2024-12-31**, realized outcome **0** — No ceasefire in 2024.
- status **completed_with_degradation**, support grade **exploratory**, latency 74.1s
- forecasts — prior_only **0.4103**, phase2_no_posterior **0.4154**, phase3_posterior **0.2462**, point_estimate **0.1385**, market **—**
- posterior: prior_mean 0.4103 → posterior_mean 0.2726 (shift -0.1376); included claims 9 → 8 effective observations; consumed True; reproducible_hash True
- **as-of audit**: bundle_hash `10aa31313f30fa01`, as_of 2024-06-01, 33 docs, included 9 / excluded 0 / suspicious 0 claims, leakage flags 0

  | doc | source | published | temporal_status |
  |---|---|---|---|
  | e8ec22f2fbac9b89 | Council on Foreign Relations | 2024-02-21 | likely_pre_asof |
  | 520172175792d3da | European Council on Foreign Relations | 2024-02-19 | likely_pre_asof |
  | 15b92175eb7f2851 | Reuters | 2024-02-13 | likely_pre_asof |
  | c2e524a153166c92 | stimson.org | 2024-02-22 | likely_pre_asof |
  | 97a1f0929e953fe3 | PBS | 2024-03-06 | likely_pre_asof |
  | 12c341968f462028 | Foreign Affairs | 2024-04-16 | likely_pre_asof |
  | 2b8d2910d33e6ec0 | The Guardian | 2024-04-22 | likely_pre_asof |
  | e262068f6e5ea900 | Atlantic Council | 2024-05-30 | likely_pre_asof |
  | 845619cde898ace0 | RAND | 2024-02-09 | likely_pre_asof |
  | 6733b645ca2c9d1e | WBUR | 2024-05-06 | likely_pre_asof |
  | 22f7d97457e333d5 | Institute for National Strategic Studies (INSS) | 2024-03-04 | likely_pre_asof |
  | ebe3b10e6509dbef | Fair Observer | 2024-02-24 | likely_pre_asof |

### `india_t20` — Will India win the 2024 ICC Men's T20 Cricket World Cup?
- domain **sports**, as_of **2024-06-20**, horizon **2024-06-29**, realized outcome **1** — India won the final 2024-06-29.
- status **completed_with_degradation**, support grade **highly_speculative**, latency 63.6s
- forecasts — prior_only **0.5000**, phase2_no_posterior **0.5211**, phase3_posterior **0.6761**, point_estimate **0.6197**, market **—**
- posterior: prior_mean 0.5000 → posterior_mean 0.5051 (shift 0.0051); included claims 10 → 8 effective observations; consumed True; reproducible_hash True
- **as-of audit**: bundle_hash `da81bdf5d5942b2f`, as_of 2024-06-20, 37 docs, included 10 / excluded 0 / suspicious 0 claims, leakage flags 0

  | doc | source | published | temporal_status |
  |---|---|---|---|
  | 3cd73682a451fb44 | USA Cricket | 2024-06-14 | likely_pre_asof |
  | e6f60068ef864840 | YouGov | 2024-04-29 | likely_pre_asof |
  | 73a3e423e8ab2057 | ICC | 2024-06-09 | likely_pre_asof |
  | 26583908ca12f225 | Al Jazeera | 2024-05-28 | likely_pre_asof |
  | fbd37d4f8bf638bf | Sling TV | 2024-06-20 | uncertain |
  | c9a763aefb98ab79 | USA Today | 2024-06-04 | likely_pre_asof |
  | 0b31f2d36c18a44c | Houston Chronicle | 2024-06-12 | likely_pre_asof |
  | e77bd7cb43d4bd2d | Al Jazeera | 2024-06-17 | likely_pre_asof |
  | 831291cdbf0fd0bc | Al Jazeera | 2024-05-29 | likely_pre_asof |
  | 078568b459dd1f1a | ICC | 2024-06-20 | uncertain |
  | c041a4b1c1bd6a6d | Al Jazeera | 2024-05-28 | likely_pre_asof |
  | 0dfe9775b0d4573b | Al Jazeera | 2024-05-30 | likely_pre_asof |

### `real_ucl` — Will Real Madrid win the 2024 UEFA Champions League final?
- domain **sports**, as_of **2024-05-25**, horizon **2024-06-01**, realized outcome **1** — Real Madrid won the final 2024-06-01.
- status **completed_with_degradation**, support grade **highly_speculative**, latency 100.6s
- forecasts — prior_only **0.5000**, phase2_no_posterior **0.5205**, phase3_posterior **0.5479**, point_estimate **0.6027**, market **—**
- posterior: prior_mean 0.5000 → posterior_mean 0.5412 (shift 0.0411); included claims 21 → 8 effective observations; consumed True; reproducible_hash True
- **as-of audit**: bundle_hash `9d6b659f83dfbde2`, as_of 2024-05-25, 37 docs, included 21 / excluded 0 / suspicious 0 claims, leakage flags 0

  | doc | source | published | temporal_status |
  |---|---|---|---|
  | adb03a1a596ffc6a | ESPN | 2024-05-09 | likely_pre_asof |
  | 97b0781cc7a03cf5 | Bleacher Report | 2024-05-08 | likely_pre_asof |
  | 84bbe6ff12f04320 | ESPN Press Room | 2024-05-10 | likely_pre_asof |
  | 7c4fb71569b093fb | Diario AS | 2024-05-09 | likely_pre_asof |
  | 866f9ac7d750cefc | USA Today | 2024-05-08 | likely_pre_asof |
  | 0f53ae29bca21dbe | FOX Sports | 2024-05-08 | likely_pre_asof |
  | 9ea925e761ddf663 | AP News | 2024-05-13 | likely_pre_asof |
  | 39994fa4366d763d | The University News | 2024-04-29 | likely_pre_asof |
  | 5b3de99302b3d282 | ESPN | 2024-05-08 | likely_pre_asof |
  | dddc2e23f8feddb6 | Bleacher Report | 2024-05-08 | likely_pre_asof |
  | b7ae008632e27b0b | USA Today | 2024-05-08 | likely_pre_asof |
  | 554a841f5ce327a7 | Diario AS | 2024-05-08 | likely_pre_asof |

### `starship_catch` — Will SpaceX catch a Starship booster with the launch tower in 2024?
- domain **science**, as_of **2024-10-01**, horizon **2024-12-31**, realized outcome **1** — Booster caught on Flight 5, 2024-10-13.
- status **harness_error**, support grade **None**, latency 212.8s
- ERROR: KeyError: 'planes'
