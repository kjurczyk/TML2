?	?0?*?9@?0?*?9@!?0?*?9@	?k?ސ2@?k?ސ2@!?k?ސ2@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?0?*?9@??ܵ.@AEGr??@Y?St$??@*	3333???@2P
Iterator::Model::Prefetch+??@!??[?*H@)+??@1??[?*H@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?*?ԉ@!????ScI@)??u??@1J??U"?A@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator??H.???!?xg?b./@)??H.???1?xg?b./@:Preprocessing2F
Iterator::Model???{??@!G@*-??H@)I.?!????1Ԏ?Nt???:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensora2U0*?S?!L?d???)a2U0*?S?1L?d???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 18.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t60.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?k?ސ2@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??ܵ.@??ܵ.@!??ܵ.@      ??!       "      ??!       *      ??!       2	EGr??@EGr??@!EGr??@:      ??!       B      ??!       J	?St$??@?St$??@!?St$??@R      ??!       Z	?St$??@?St$??@!?St$??@JCPU_ONLYY?k?ސ2@b Y      Y@q!??L?N@"?	
both?Your program is MODERATELY input-bound because 18.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t60.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.:
Refer to the TF2 Profiler FAQb?60.1617% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 