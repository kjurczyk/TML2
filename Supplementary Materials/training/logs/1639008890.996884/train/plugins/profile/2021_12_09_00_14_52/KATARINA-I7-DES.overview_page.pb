?	?c]?F?@?c]?F?@!?c]?F?@	?9I??Q@?9I??Q@!?9I??Q@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?c]?F?@??ǘ????Alxz?,C??Y?q??[@*	3333?̻@2P
Iterator::Model::Prefetch??z6?@!.D	j-I@)??z6?@1.D	j-I@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap???H?@!?wK?IOH@)??Pk?w@17c??S?B@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?:pΈ???!???i??%@)?:pΈ???1???i??%@:Preprocessing2F
Iterator::Model?m4??@@!????I@)㥛? ???1?zhui??:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor????MbP?!6??ǌ?)????MbP?16??ǌ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 71.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2s3.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?9I??Q@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??ǘ??????ǘ????!??ǘ????      ??!       "      ??!       *      ??!       2	lxz?,C??lxz?,C??!lxz?,C??:      ??!       B      ??!       J	?q??[@?q??[@!?q??[@R      ??!       Z	?q??[@?q??[@!?q??[@JCPU_ONLYY?9I??Q@b Y      Y@q?????S@"?	
host?Your program is HIGHLY input-bound because 71.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nomoderate"s3.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.:
Refer to the TF2 Profiler FAQb?79.3908% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 