[10/Dec/2023 16:38:59] INFO - label_list: ['0', '1']
Loading BERT tokenizer...
[10/Dec/2023 16:39:03] INFO - device: cuda n_gpu: 2, distributed training: False, 16-bits training: False
[10/Dec/2023 16:39:03] INFO - load train tsv.
[10/Dec/2023 16:39:03] INFO - Writing example 0 of 31296
[10/Dec/2023 16:39:03] INFO - *** Example ***
[10/Dec/2023 16:39:03] INFO - number of examples: 31296
[10/Dec/2023 16:39:03] INFO - guid: train-0
[10/Dec/2023 16:39:03] INFO - tokens: [CLS] acquired abnormal ##ity [SEP] location of [SEP] experimental model of disease [SEP]
[10/Dec/2023 16:39:03] INFO - input_ids: 101 2888 22832 1785 102 2450 1104 102 6700 2235 1104 3653 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
[10/Dec/2023 16:39:03] INFO - input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
[10/Dec/2023 16:39:03] INFO - segment_ids: 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
[10/Dec/2023 16:39:03] INFO - label: 1 (id = 1)
[10/Dec/2023 16:39:04] INFO - Writing example 10000 of 31296
[10/Dec/2023 16:39:06] INFO - Writing example 20000 of 31296
[10/Dec/2023 16:39:08] INFO - Writing example 30000 of 31296
[10/Dec/2023 16:39:08] INFO - avg_len: 10.0
[10/Dec/2023 16:39:08] INFO - ***** Running training *****
[10/Dec/2023 16:39:08] INFO -   Batch size = 128
Training loss:  166.84965753555298 31296
[10/Dec/2023 16:41:40] INFO - load train tsv.
[10/Dec/2023 16:41:40] INFO - Writing example 0 of 652
[10/Dec/2023 16:41:40] INFO - *** Example ***
[10/Dec/2023 16:41:40] INFO - number of examples: 652
[10/Dec/2023 16:41:40] INFO - guid: dev-0
[10/Dec/2023 16:41:40] INFO - tokens: [CLS] nuclei ##c acid n ##uc ##leo ##side or n ##uc ##leo ##tide [SEP] affects [SEP] mental or behavioral d ##ys ##function [SEP]
[10/Dec/2023 16:41:40] INFO - input_ids: 101 27349 1665 5190 183 21977 26918 5570 1137 183 21977 26918 23767 102 13974 102 4910 1137 18560 173 6834 26420 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
[10/Dec/2023 16:41:40] INFO - input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
[10/Dec/2023 16:41:40] INFO - segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
[10/Dec/2023 16:41:40] INFO - label: 1 (id = 1)
[10/Dec/2023 16:41:40] INFO - avg_len: 9.0
[10/Dec/2023 16:41:41] INFO - ***** Running training *****
[10/Dec/2023 16:41:41] INFO -   Batch size = 128
[10/Dec/2023 16:41:42] INFO - *** Example Eval ***
[10/Dec/2023 16:41:42] INFO - preds: [0 0 0 0 0 0 0 0 0 1]
[10/Dec/2023 16:41:42] INFO - labels: [1 1 1 1 1 1 1 1 1 1]
[10/Dec/2023 16:41:42] INFO - ***** Eval results *****
[10/Dec/2023 16:41:42] INFO -   acc = 0.18098159509202455
[10/Dec/2023 16:41:42] INFO -   eval_loss = 0.7520643671353658
[10/Dec/2023 16:41:42] INFO -   global_step = 0
[10/Dec/2023 16:41:42] INFO -   loss = 0.6810190103491959
timebudget report per main cycle...
                     main: 100.0%  162638.45ms/cyc @     1.0 calls/cyc
               train_loop:  91.5%  148829.54ms/cyc @     1.0 calls/cyc
convert_examples_to_features:   3.5%  5660.76ms/cyc @     2.0 calls/cyc
                eval_loop:   0.8%  1242.51ms/cyc @     1.0 calls/cyc
                 get_data:   0.5%   744.08ms/cyc @     2.0 calls/cyc
         _create_examples:   0.1%   152.62ms/cyc @     2.0 calls/cyc
                _read_tsv:   0.0%    41.09ms/cyc @     2.0 calls/cyc
timebudget report...
                     main: 162638.45ms for      1 calls
               train_loop: 148829.54ms for      1 calls
convert_examples_to_features: 2830.38ms for      2 calls
                eval_loop: 1242.51ms for      1 calls
                 get_data:  372.04ms for      2 calls
         _create_examples:   76.31ms for      2 calls
                _read_tsv:   20.55ms for      2 calls
timebudget report...
                     main: 162638.45ms for      1 calls
               train_loop: 148829.54ms for      1 calls
convert_examples_to_features: 2830.38ms for      2 calls
                eval_loop: 1242.51ms for      1 calls
                 get_data:  372.04ms for      2 calls
         _create_examples:   76.31ms for      2 calls
                _read_tsv:   20.55ms for      2 calls
