| méthode | langue | MRR avant | MRR après | R@3 avant | R@3 après |
|---|---|---|---|---|---|
| bm25 | fr | 0.42 | 0.42 | 0.47 | 0.47 |
| bm25 | en | 0.29 | 0.29 | 0.20 | 0.20 |
| bm25 | zh | 0.00 | 0.00 | 0.00 | 0.00 |
| bm25 | toutes | 0.29 | 0.29 | 0.28 | 0.28 |
| dense | fr | 0.65 | 0.78 | 0.73 | 0.83 |
| dense | en | 0.28 | 0.51 | 0.15 | 0.50 |
| dense | zh | 0.46 | 0.60 | 0.50 | 0.60 |
| dense | toutes | 0.51 | 0.67 | 0.53 | 0.69 |
| hybride | fr | 0.62 | 0.69 | 0.65 | 0.65 |
| hybride | en | 0.46 | 0.52 | 0.50 | 0.60 |
| hybride | zh | 0.46 | 0.60 | 0.50 | 0.60 |
| hybride | toutes | 0.54 | 0.62 | 0.57 | 0.62 |
| hybride+rerank | fr | 0.82 | 0.85 | 0.76 | 0.79 |
| hybride+rerank | en | 0.56 | 0.67 | 0.60 | 0.70 |
| hybride+rerank | zh | 0.71 | 0.74 | 0.73 | 0.67 |
| hybride+rerank | toutes | 0.73 | 0.78 | 0.71 | 0.74 |

Questions améliorées (dense) : id, langue, rang avant -> après
  q001 fr 8 -> 1  
  q003 fr 2 -> 1
  q004 fr 17 -> 10
  q005 fr 6 -> 1
  q010 fr 3 -> 1
  q014 fr 3 -> 1
  q016 fr 999 -> 19
  q018 fr 2 -> 1
  q021 en 4 -> 1
  q023 en 2 -> 1
  q024 en 21 -> 1
  q025 en 17 -> 3
  q028 en 4 -> 1
  q031 zh 22 -> 11
  q033 zh 13 -> 12
  q034 zh 13 -> 4
  q035 zh 2 -> 1
  q038 zh 2 -> 1
  q039 zh 4 -> 1

Questions dégradées (dense) :
  q011 fr 1 -> 2
  q015 fr 17 -> 23
  q017 fr 1 -> 3
  q019 fr 1 -> 2
  q022 en 4 -> 5
  q027 en 1 -> 3
  q029 en 9 -> 999
  q030 en 17 -> 999
  q032 zh 1 -> 2
  q040 zh 9 -> 15
