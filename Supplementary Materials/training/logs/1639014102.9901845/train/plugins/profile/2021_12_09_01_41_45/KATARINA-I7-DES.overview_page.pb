?	?,C?b@?,C?b@!?,C?b@	J?'J?4N@J?'J?4N@!J?'J?4N@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?,C?b@?٬?\m??A??????@Y????<?@*	gfff???@2P
Iterator::Model::PrefetchyX?5ͻ@! ????I@)yX?5ͻ@1 ????I@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?????@!F?*71=H@)8gDio@1&?e>?B@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generatorsh??|???!돏?H%@)sh??|???1돏?H%@:Preprocessing2F
Iterator::Model9EGr??@!?????I@)??ݓ????1I?,}?Z??:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensora2U0*?S?!?*?
q???)a2U0*?S?1?*?
q???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 60.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9J?'J?4N@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?٬?\m???٬?\m??!?٬?\m??      ??!       "      ??!       *      ??!       2	??????@??????@!??????@:      ??!       B      ??!       J	????<?@????<?@!????<?@R      ??!       Z	????<?@????<?@!????<?@JCPU_ONLYYJ?'J?4N@b Y      Y@q	+?a?R@"?
host?Your program is HIGHLY input-bound because 60.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?74.6466% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 