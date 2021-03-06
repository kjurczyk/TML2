?	Y?? ?(@Y?? ?(@!Y?? ?(@	i??D?YB@i??D?YB@!i??D?YB@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Y?? ?(@C??6??An???@Ya2U0*)@*	????L?@2P
Iterator::Model::Prefetch??St$?@!Si2Ԭ?G@)??St$?@1Si2Ԭ?G@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?????@!?={*?I@)~??k	?@1ʍxx?.@@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::GeneratorbX9????!U???j?2@)bX9????1U???j?2@:Preprocessing2F
Iterator::Model?HP?@!1??nH@)???~?:??1?K*????:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor/n??R?!"?E%?T??)/n??R?1"?E%?T??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 36.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9i??D?YB@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	C??6??C??6??!C??6??      ??!       "      ??!       *      ??!       2	n???@n???@!n???@:      ??!       B      ??!       J	a2U0*)@a2U0*)@!a2U0*)@R      ??!       Z	a2U0*)@a2U0*)@!a2U0*)@JCPU_ONLYYi??D?YB@b Y      Y@q?/׆`?Q@"?
host?Your program is HIGHLY input-bound because 36.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?70.8965% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 