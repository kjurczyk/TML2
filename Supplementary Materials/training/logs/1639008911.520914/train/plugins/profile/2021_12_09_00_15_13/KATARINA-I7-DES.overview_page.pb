?	??@??@!??@	u???Q@u???Q@!u???Q@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??@?V-??A?y?):???Y?y?):?@*	gfff?4?@2P
Iterator::Model::Prefetch?:pΈ?@!?Of?g?H@)?:pΈ?@1?Of?g?H@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap;?O??n@!??????H@)H?z??@1???F8C@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator&S????!0?\? z%@)&S????10?\? z%@:Preprocessing2F
Iterator::Model	?cn@!sY%_gI@)??9#J{??1?e¯?}??:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor/n??R?!?g?u?!??)/n??R?1?g?u?!??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 71.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9u???Q@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?V-???V-??!?V-??      ??!       "      ??!       *      ??!       2	?y?):????y?):???!?y?):???:      ??!       B      ??!       J	?y?):?@?y?):?@!?y?):?@R      ??!       Z	?y?):?@?y?):?@!?y?):?@JCPU_ONLYYu???Q@b Y      Y@q??'???S@"?
host?Your program is HIGHLY input-bound because 71.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?78.508% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 