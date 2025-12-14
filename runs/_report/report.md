# Отчет по экспериментам: LeNet-5 + TensorBoard (вариант 9)

## Теоретическое обоснование
- Мониторинг loss/accuracy показывает динамику качества, но не объясняет причины проблем.
- Гистограммы весов и градиентов позволяют диагностировать exploding/vanishing gradients и деградацию обучения.
- Гистограммы активаций по слоям выявляют насыщение (например, tanh уходит в ±1) и “мертвые” слои.
- Отслеживание дельты весов помогает понять, происходит ли реальное обучение или модель “застыла”.
- ReduceLROnPlateau и LR-schedule ускоряют сходимость и повышают стабильность, early stopping предотвращает переобучение.
- Confusion matrix и примеры предсказаний дают интерпретируемый срез ошибок модели.

## Постановка эксперимента
- Датасет: MNIST, 28×28 → padding до 32×32, канал 1.
- Архитектура: LeNet-5 (Conv→AvgPool→Conv→AvgPool→FC→FC→Softmax).
- Логирование: TensorBoard (scalars + histograms) + custom callbacks (gradients, activations, deltas, confusion matrix, examples).
- Сломанные сценарии: высокий LR, плохая инициализация, отсутствие нормализации.
- Стратегии LR: fixed, step decay, cyclical (triangular).
- Профилирование: время эпохи и память при разных batch_size; CPU↔GPU через отдельные процессы.

## Сводные результаты
- Таблица: `runs_summary.csv`

## Сравнительный анализ
```text
Топ запусков по test_accuracy:
          run_name lr_strategy  test_accuracy  best_val_accuracy  avg_epoch_time_sec  rss_peak_mb
  compare_cyclical    cyclical         0.9877           0.987667            3.243760  1677.035156
  compare_cyclical    cyclical         0.9877           0.987667            2.818743  1678.816406
  compare_cyclical    cyclical         0.9877           0.987667            2.874990  1676.062500
    broken_no_norm       fixed         0.9867           0.987167            1.952181  1613.105469
    broken_no_norm       fixed         0.9867           0.987167            1.712022  1613.703125
    broken_no_norm       fixed         0.9867           0.987167            1.843508  1614.503906
    broken_no_norm       fixed         0.9867           0.987167            1.819445  1612.574219
compare_step_decay  step_decay         0.9858           0.987000            1.548318  1650.714844
compare_step_decay  step_decay         0.9858           0.987000            1.692023  1649.898438
compare_step_decay  step_decay         0.9858           0.987000            1.706978  1648.257812
```

## Рекомендации по гиперпараметрам
- Всегда нормализуйте входные данные (например, MNIST: деление на 255). Без нормализации ухудшается сходимость и стабильность градиентов.
- Используйте устойчивые инициализации (Glorot/He). Слишком “широкая” RandomNormal может провоцировать нестабильные активации/градиенты и замедлять обучение.
- Слишком высокий learning rate часто приводит к колебаниям loss/accuracy или к NaN. Начинайте с 1e-3 (Adam) и включайте ReduceLROnPlateau / scheduler.
- По результатам сравнения, наиболее эффективная стратегия (test_accuracy максимум): compare_cyclical (strategy=cyclical). Рекомендуется использовать её как дефолт для похожих задач.
- Для ускорения выхода на плато используйте: ReduceLROnPlateau + сохранение best weights (ModelCheckpoint) + early stopping.
- Средняя корреляция между изменением весов и val_accuracy (по доступным прогонам): -0.827. Используйте этот сигнал как индикатор: если веса почти не меняются при низкой точности — вероятно vanishing gradients/слишком маленький LR.

## Как быстро развернуть аналогичный эксперимент
- Используйте `experiment_template.json` из нужного run_dir как стартовую конфигурацию.
- Меняйте: `lr.strategy`, `lr.lr/base_lr/max_lr`, `train.batch_size/epochs`, включайте/выключайте callbacks.

## Где смотреть визуализации
- Запустите TensorBoard: `tensorboard --logdir runs`
- Секции: Scalars (loss/accuracy/perf), Histograms (weights/grads/activations), Images (confusion matrix / examples), Graph.
