?	???@???@!???@	??1\@??1\@!??1\@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???@tF??_??A?6?[?@Y?;Nё\??*	33333C?@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generatora??+e??!???.AW@)a??+e??1???.AW@:Preprocessing2F
Iterator::Model??\m????!?[???d@)B>?٬???1~?ś?^@:Preprocessing2P
Iterator::Model::Prefetch/n????!K!z????)/n????1K!z????:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?9#J{???!C???iW@)???_vOn?1?>գ?B??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??1\@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	tF??_??tF??_??!tF??_??      ??!       "      ??!       *      ??!       2	?6?[?@?6?[?@!?6?[?@:      ??!       B      ??!       J	?;Nё\???;Nё\??!?;Nё\??R      ??!       Z	?;Nё\???;Nё\??!?;Nё\??JCPU_ONLYY??1\@b Y      Y@q6?;?ZOW@"?
both?Your program is POTENTIALLY input-bound because 3.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?93.2399% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 