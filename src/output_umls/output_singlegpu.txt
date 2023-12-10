[10/Dec/2023 16:57:39] INFO - label_list: ['0', '1']
Loading BERT tokenizer...
[10/Dec/2023 16:57:42] INFO - device: cuda n_gpu: 2, distributed training: False, 16-bits training: False
[10/Dec/2023 16:57:42] INFO - load train tsv.
[10/Dec/2023 16:57:42] INFO - Writing example 0 of 31296
[10/Dec/2023 16:57:42] INFO - *** Example ***
[10/Dec/2023 16:57:42] INFO - number of examples: 31296
[10/Dec/2023 16:57:42] INFO - guid: train-0
[10/Dec/2023 16:57:42] INFO - tokens: [CLS] acquired abnormal ##ity [SEP] location of [SEP] experimental model of disease [SEP]
[10/Dec/2023 16:57:42] INFO - input_ids: 101 2888 22832 1785 102 2450 1104 102 6700 2235 1104 3653 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
[10/Dec/2023 16:57:42] INFO - input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
[10/Dec/2023 16:57:42] INFO - segment_ids: 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
[10/Dec/2023 16:57:42] INFO - label: 1 (id = 1)
[10/Dec/2023 16:57:44] INFO - Writing example 10000 of 31296
[10/Dec/2023 16:57:46] INFO - Writing example 20000 of 31296
[10/Dec/2023 16:57:47] INFO - Writing example 30000 of 31296
[10/Dec/2023 16:57:48] INFO - avg_len: 10.0
[10/Dec/2023 16:57:48] INFO - ***** Running training *****
[10/Dec/2023 16:57:48] INFO -   Batch size = 128
Training loss:  215.17446100711823 31296
[10/Dec/2023 17:00:17] INFO - load train tsv.
[10/Dec/2023 17:00:17] INFO - Writing example 0 of 652
[10/Dec/2023 17:00:17] INFO - *** Example ***
[10/Dec/2023 17:00:17] INFO - number of examples: 652
[10/Dec/2023 17:00:17] INFO - guid: dev-0
[10/Dec/2023 17:00:17] INFO - tokens: [CLS] nuclei ##c acid n ##uc ##leo ##side or n ##uc ##leo ##tide [SEP] affects [SEP] mental or behavioral d ##ys ##function [SEP]
[10/Dec/2023 17:00:17] INFO - input_ids: 101 27349 1665 5190 183 21977 26918 5570 1137 183 21977 26918 23767 102 13974 102 4910 1137 18560 173 6834 26420 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
[10/Dec/2023 17:00:17] INFO - input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
[10/Dec/2023 17:00:17] INFO - segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
[10/Dec/2023 17:00:17] INFO - label: 1 (id = 1)
[10/Dec/2023 17:00:18] INFO - avg_len: 9.0
[10/Dec/2023 17:00:19] INFO - ***** Running training *****
[10/Dec/2023 17:00:19] INFO -   Batch size = 128
[10/Dec/2023 17:00:20] INFO - *** Example Eval ***
[10/Dec/2023 17:00:20] INFO - preds: [1 1 1 1 1 1 1 1 1 1]
[10/Dec/2023 17:00:20] INFO - labels: [1 1 1 1 1 1 1 1 1 1]
[10/Dec/2023 17:00:20] INFO - ***** Eval results *****
[10/Dec/2023 17:00:20] INFO -   acc = 1.0
[10/Dec/2023 17:00:20] INFO -   eval_loss = 0.5078530311584473
[10/Dec/2023 17:00:20] INFO -   global_step = 0
[10/Dec/2023 17:00:20] INFO -   loss = 0.8782631061515029
timebudget report per main cycle...
                     main: 100.0%  161243.32ms/cyc @     1.0 calls/cyc
               train_loop:  91.2%  147108.75ms/cyc @     1.0 calls/cyc
convert_examples_to_features:   3.5%  5626.06ms/cyc @     2.0 calls/cyc
                eval_loop:   0.7%  1111.48ms/cyc @     1.0 calls/cyc
                 get_data:   0.5%   742.92ms/cyc @     2.0 calls/cyc
         _create_examples:   0.1%   179.04ms/cyc @     2.0 calls/cyc
                _read_tsv:   0.0%    28.98ms/cyc @     2.0 calls/cyc
timebudget report...
                     main: 161243.32ms for      1 calls
               train_loop: 147108.75ms for      1 calls
convert_examples_to_features: 2813.03ms for      2 calls
                eval_loop: 1111.48ms for      1 calls
                 get_data:  371.46ms for      2 calls
         _create_examples:   89.52ms for      2 calls
                _read_tsv:   14.49ms for      2 calls
timebudget report...
                     main: 161243.32ms for      1 calls
               train_loop: 147108.75ms for      1 calls
convert_examples_to_features: 2813.03ms for      2 calls
                eval_loop: 1111.48ms for      1 calls
                 get_data:  371.46ms for      2 calls
         _create_examples:   89.52ms for      2 calls
                _read_tsv:   14.49ms for      2 calls
