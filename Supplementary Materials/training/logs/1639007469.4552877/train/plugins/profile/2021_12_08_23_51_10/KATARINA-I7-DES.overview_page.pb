?	&䃞?*"@&䃞?*"@!&䃞?*"@	y?r?E@y?r?E@!y?r?E@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$&䃞?*"@=?U????A
h"lx?@Y?q???@*	fffff.?@2P
Iterator::Model::Prefetch??S??@!_????wG@)??S??@1_????wG@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?C???@@!c???1J@)?|гYu@1hrw?E?@@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generatorh??s???!?_+2@)h??s???1?_+2@:Preprocessing2F
Iterator::Modelڬ?\m?@!?q*??G@)b??4?8??1?'?$???:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor-C??6J?!?<?P?ƃ?)-C??6J?1?<?P?ƃ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 43.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9y?r?E@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	=?U????=?U????!=?U????      ??!       "      ??!       *      ??!       2	
h"lx?@
h"lx?@!
h"lx?@:      ??!       B      ??!       J	?q???@?q???@!?q???@R      ??!       Z	?q???@?q???@!?q???@JCPU_ONLYYy?r?E@b Y      Y@q*?t?85R@"?
host?Your program is HIGHLY input-bound because 43.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?72.8316% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 