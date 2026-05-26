## Данные об авторе
Цыбин Егор

cybin_ea_23

3 Курс | 6 Семестр

Кибербезопаснсоть

Курсовой проект

# Social Network Analysis

Курсовой проект: анализ графа сетевого взаимодействия пользователей —
выделение сообществ, выявление лидеров мнений и изолированных групп
классическими алгоритмами и графовыми нейронными сетями.

## Требования

- Python **3.12+**
- [uv](https://docs.astral.sh/uv/) для управления зависимостями

## Установка

```bash
uv sync
```

Виртуальное окружение и lock-файл создаются автоматически. Все зависимости
зафиксированы в `uv.lock`.

## Запуск пайплайна

```bash
uv run python main.py
```

Поведение полностью управляется файлом `config.toml`: список датасетов,
включённые шаги, гиперпараметры алгоритмов и GNN, уровень логирования.

```toml
log_level = "normal"   # quiet | normal | verbose | debug

[datasets]
include = ["karate", "dolphins", "football",
           "email-Eu-core", "facebook-ego-1684", "dblp-top500"]

[steps]
structural = true
benchmark  = true
gnn        = true
leaders    = true
isolated   = true
visualize  = true

[gnn]
models           = ["gcn", "graphsage"]
epochs           = 500
patience         = 50
checkpoint_dir   = "models"
force_retrain    = false   # true — игнорировать кэш и переобучить
```

Большие датасеты скачиваются автоматически в `data/raw/` при первом запуске.
Графики и CSV-таблицы — в `results/`. Чекпоинты обученных GNN — в `models/`,
повторный запуск использует их без переобучения.

## Структура проекта

```
config.toml              конфигурация
main.py                  тонкий entrypoint
src/
  config.py              загрузка config.toml
  log.py                 логирование с уровнями
  datasets.py            единый реестр всех датасетов
  pipeline.py            шаги пайплайна (structural, benchmark, gnn, ...)
  benchmark.py           бенчмарк алгоритмов на сетке датасетов
  community.py           Louvain, Leiden, Label Propagation, Girvan–Newman, Infomap, Walktrap
  centrality.py          5 метрик + композитный ранг
  isolation.py           изолированные группы (conductance, k-core)
  gnn.py                 GCN, GraphSAGE с save/load чекпоинтов
  graph_stats.py         структурные характеристики
  graph_models.py        случайные графы (ER, BA, WS)
  metrics.py             NMI, AMI, ARI, purity, accuracy, modularity
  visualization.py       графики сообществ и распределения степеней
tests/                   pytest
results/                 CSV-таблицы и PNG-графики (генерируются)
models/                  чекпоинты GNN (генерируются)
data/raw/                кэш скачанных датасетов
```

## Датасеты

| Имя | n | m | k | Источник |
|-----|---|---|---|----------|
| karate | 34 | 78 | 2 | Zachary 1977 |
| dolphins | 62 | 159 | 2 | Lusseau et al. 2003 |
| football | 115 | 613 | 12 | Girvan & Newman 2002 |
| email-Eu-core | ~1000 | ~16k | 42 | SNAP |
| facebook-ego-1684 | ~800 | ~15k | 18 | SNAP |
| dblp-top500 | ~600 | ~3k | 500 | SNAP |

## Добавление своего алгоритма

```python
from src.benchmark import benchmark
from src.datasets import load_all

def my_algo(g):   # igraph.Graph -> np.ndarray меток
    ...

benchmark({"my_algo": my_algo}, datasets=load_all())
```

## Тесты

```bash
uv run pytest
```

