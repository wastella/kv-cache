/*
 * Real next-token probabilities from GPT-2 (124M params, "gpt2" on
 * Hugging Face) — an actual forward pass over three sentences below, not
 * simulated. For every token, `steps[i]` gives the model's real
 * probability for the token that actually came next, its rank among all
 * ~50K vocab entries, and its top-3 guesses. `perplexity` is
 * exp(mean(-ln actualProb)) computed directly from those numbers — not
 * reverse-engineered from a published constant. Regenerate with
 * tools/gen_gpt2_perplexity.py.
 */
const GPT2_PPX_SENTENCES = [
  {
    "id": "iconic",
    "label": "iconic phrase",
    "text": "The quick brown fox jumps over the lazy dog.",
    "tokens": [
      "The",
      " quick",
      " brown",
      " fox",
      " jumps",
      " over",
      " the",
      " lazy",
      " dog",
      "."
    ],
    "steps": [
      null,
      {
        "surprisal": 8.7124,
        "actualProb": 0.000165,
        "rank": 971,
        "top": [
          {
            "tok": "\n",
            "p": 0.0136
          },
          {
            "tok": " the",
            "p": 0.009974
          },
          {
            "tok": " \"",
            "p": 0.008197
          }
        ]
      },
      {
        "surprisal": 8.9043,
        "actualProb": 0.000136,
        "rank": 756,
        "top": [
          {
            "tok": "-",
            "p": 0.074066
          },
          {
            "tok": " answer",
            "p": 0.066139
          },
          {
            "tok": " and",
            "p": 0.060664
          }
        ]
      },
      {
        "surprisal": 6.2615,
        "actualProb": 0.001908,
        "rank": 43,
        "top": [
          {
            "tok": "ie",
            "p": 0.192089
          },
          {
            "tok": "ies",
            "p": 0.040143
          },
          {
            "tok": "-",
            "p": 0.033384
          }
        ]
      },
      {
        "surprisal": 4.8561,
        "actualProb": 0.007781,
        "rank": 17,
        "top": [
          {
            "tok": "es",
            "p": 0.129673
          },
          {
            "tok": " was",
            "p": 0.045685
          },
          {
            "tok": " is",
            "p": 0.034406
          }
        ]
      },
      {
        "surprisal": 2.7479,
        "actualProb": 0.064063,
        "rank": 5,
        "top": [
          {
            "tok": " up",
            "p": 0.168194
          },
          {
            "tok": " out",
            "p": 0.137823
          },
          {
            "tok": " on",
            "p": 0.076903
          }
        ]
      },
      {
        "surprisal": 0.8609,
        "actualProb": 0.422791,
        "rank": 1,
        "top": [
          {
            "tok": " the",
            "p": 0.422791
          },
          {
            "tok": " a",
            "p": 0.107238
          },
          {
            "tok": " to",
            "p": 0.055804
          }
        ]
      },
      {
        "surprisal": 7.249,
        "actualProb": 0.000711,
        "rank": 196,
        "top": [
          {
            "tok": " fence",
            "p": 0.068387
          },
          {
            "tok": " edge",
            "p": 0.033484
          },
          {
            "tok": " wall",
            "p": 0.021257
          }
        ]
      },
      {
        "surprisal": 3.8173,
        "actualProb": 0.021986,
        "rank": 5,
        "top": [
          {
            "tok": ",",
            "p": 0.055662
          },
          {
            "tok": " fox",
            "p": 0.042476
          },
          {
            "tok": " brown",
            "p": 0.033832
          }
        ]
      },
      {
        "surprisal": 2.4051,
        "actualProb": 0.09026,
        "rank": 3,
        "top": [
          {
            "tok": " and",
            "p": 0.320523
          },
          {
            "tok": ",",
            "p": 0.185626
          },
          {
            "tok": ".",
            "p": 0.09026
          }
        ]
      }
    ],
    "perplexity": 162.4701
  },
  {
    "id": "technical",
    "label": "technical sentence",
    "text": "The transformer stores a key and value vector for every token.",
    "tokens": [
      "The",
      " transformer",
      " stores",
      " a",
      " key",
      " and",
      " value",
      " vector",
      " for",
      " every",
      " token",
      "."
    ],
    "steps": [
      null,
      {
        "surprisal": 12.3976,
        "actualProb": 4e-06,
        "rank": 23589,
        "top": [
          {
            "tok": "\n",
            "p": 0.0136
          },
          {
            "tok": " the",
            "p": 0.009974
          },
          {
            "tok": " \"",
            "p": 0.008197
          }
        ]
      },
      {
        "surprisal": 9.2395,
        "actualProb": 9.7e-05,
        "rank": 706,
        "top": [
          {
            "tok": " is",
            "p": 0.221661
          },
          {
            "tok": " will",
            "p": 0.042782
          },
          {
            "tok": " was",
            "p": 0.039358
          }
        ]
      },
      {
        "surprisal": 2.5402,
        "actualProb": 0.078852,
        "rank": 2,
        "top": [
          {
            "tok": " the",
            "p": 0.295121
          },
          {
            "tok": " a",
            "p": 0.078852
          },
          {
            "tok": " energy",
            "p": 0.031482
          }
        ]
      },
      {
        "surprisal": 6.9354,
        "actualProb": 0.000973,
        "rank": 153,
        "top": [
          {
            "tok": " voltage",
            "p": 0.036428
          },
          {
            "tok": " number",
            "p": 0.030426
          },
          {
            "tok": " single",
            "p": 0.026027
          }
        ]
      },
      {
        "surprisal": 3.1064,
        "actualProb": 0.044763,
        "rank": 5,
        "top": [
          {
            "tok": " to",
            "p": 0.116564
          },
          {
            "tok": " that",
            "p": 0.079395
          },
          {
            "tok": " in",
            "p": 0.072743
          }
        ]
      },
      {
        "surprisal": 4.0758,
        "actualProb": 0.016979,
        "rank": 7,
        "top": [
          {
            "tok": " a",
            "p": 0.203327
          },
          {
            "tok": " the",
            "p": 0.124468
          },
          {
            "tok": " an",
            "p": 0.046395
          }
        ]
      },
      {
        "surprisal": 6.573,
        "actualProb": 0.001398,
        "rank": 54,
        "top": [
          {
            "tok": " in",
            "p": 0.091695
          },
          {
            "tok": ",",
            "p": 0.082286
          },
          {
            "tok": " for",
            "p": 0.075751
          }
        ]
      },
      {
        "surprisal": 2.7442,
        "actualProb": 0.0643,
        "rank": 5,
        "top": [
          {
            "tok": ",",
            "p": 0.137694
          },
          {
            "tok": ".",
            "p": 0.105571
          },
          {
            "tok": " that",
            "p": 0.075159
          }
        ]
      },
      {
        "surprisal": 4.1581,
        "actualProb": 0.015637,
        "rank": 6,
        "top": [
          {
            "tok": " the",
            "p": 0.315654
          },
          {
            "tok": " each",
            "p": 0.300471
          },
          {
            "tok": " a",
            "p": 0.055872
          }
        ]
      },
      {
        "surprisal": 6.8562,
        "actualProb": 0.001053,
        "rank": 132,
        "top": [
          {
            "tok": " element",
            "p": 0.091775
          },
          {
            "tok": " key",
            "p": 0.030937
          },
          {
            "tok": " value",
            "p": 0.028571
          }
        ]
      },
      {
        "surprisal": 1.9792,
        "actualProb": 0.13818,
        "rank": 2,
        "top": [
          {
            "tok": " in",
            "p": 0.16466
          },
          {
            "tok": ".",
            "p": 0.13818
          },
          {
            "tok": " that",
            "p": 0.078131
          }
        ]
      }
    ],
    "perplexity": 247.05
  },
  {
    "id": "nonsense",
    "label": "word salad",
    "text": "Purple elephants calculate Tuesday inside the mathematics of jazz.",
    "tokens": [
      "Pur",
      "ple",
      " elephants",
      " calculate",
      " Tuesday",
      " inside",
      " the",
      " mathematics",
      " of",
      " jazz",
      "."
    ],
    "steps": [
      null,
      {
        "surprisal": 7.3017,
        "actualProb": 0.000674,
        "rank": 178,
        "top": [
          {
            "tok": ".",
            "p": 0.020049
          },
          {
            "tok": "-",
            "p": 0.019879
          },
          {
            "tok": ",",
            "p": 0.019846
          }
        ]
      },
      {
        "surprisal": 12.8351,
        "actualProb": 3e-06,
        "rank": 19010,
        "top": [
          {
            "tok": "-",
            "p": 0.027138
          },
          {
            "tok": ",",
            "p": 0.020045
          },
          {
            "tok": ")",
            "p": 0.018454
          }
        ]
      },
      {
        "surprisal": 12.1797,
        "actualProb": 5e-06,
        "rank": 5527,
        "top": [
          {
            "tok": ",",
            "p": 0.114133
          },
          {
            "tok": " are",
            "p": 0.08348
          },
          {
            "tok": ".",
            "p": 0.058144
          }
        ]
      },
      {
        "surprisal": 12.6928,
        "actualProb": 3e-06,
        "rank": 8944,
        "top": [
          {
            "tok": " the",
            "p": 0.27231
          },
          {
            "tok": " their",
            "p": 0.164775
          },
          {
            "tok": " a",
            "p": 0.043409
          }
        ]
      },
      {
        "surprisal": 10.203,
        "actualProb": 3.7e-05,
        "rank": 482,
        "top": [
          {
            "tok": "'s",
            "p": 0.566627
          },
          {
            "tok": " morning",
            "p": 0.049216
          },
          {
            "tok": " night",
            "p": 0.041576
          }
        ]
      },
      {
        "surprisal": 0.9644,
        "actualProb": 0.381218,
        "rank": 1,
        "top": [
          {
            "tok": " the",
            "p": 0.381218
          },
          {
            "tok": " a",
            "p": 0.264776
          },
          {
            "tok": " their",
            "p": 0.080131
          }
        ]
      },
      {
        "surprisal": 12.3496,
        "actualProb": 4e-06,
        "rank": 13947,
        "top": [
          {
            "tok": " zoo",
            "p": 0.022826
          },
          {
            "tok": " National",
            "p": 0.009529
          },
          {
            "tok": " city",
            "p": 0.008192
          }
        ]
      },
      {
        "surprisal": 3.6983,
        "actualProb": 0.024765,
        "rank": 6,
        "top": [
          {
            "tok": " department",
            "p": 0.165167
          },
          {
            "tok": " lab",
            "p": 0.150976
          },
          {
            "tok": " classroom",
            "p": 0.086719
          }
        ]
      },
      {
        "surprisal": 9.4643,
        "actualProb": 7.8e-05,
        "rank": 1547,
        "top": [
          {
            "tok": " the",
            "p": 0.175855
          },
          {
            "tok": " a",
            "p": 0.076243
          },
          {
            "tok": " their",
            "p": 0.018516
          }
        ]
      },
      {
        "surprisal": 1.2573,
        "actualProb": 0.284433,
        "rank": 1,
        "top": [
          {
            "tok": ".",
            "p": 0.284433
          },
          {
            "tok": ",",
            "p": 0.150893
          },
          {
            "tok": " and",
            "p": 0.074626
          }
        ]
      }
    ],
    "perplexity": 4002.234
  }
];
