?	??????@??????@!??????@	?? ],R@?? ],R@!?? ],R@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??????@?ZӼ???A??	h"??Y>yX?5M@*	????L??@2P
Iterator::Model::Prefetch{?/L??@!o0?A?9I@){?/L??@1o0?A?9I@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapm????R@!??n^?/H@)j?q???@18W?^?C@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::GeneratorV-???!,3ʥ??"@)V-???1,3ʥ??"@:Preprocessing2F
Iterator::Model???~?:@!x??^?I@){?/L?
??1I?(?K???:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor??H?}M?!?????.??)??H?}M?1?????.??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 72.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2s3.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?? ],R@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?ZӼ????ZӼ???!?ZӼ???      ??!       "      ??!       *      ??!       2	??	h"????	h"??!??	h"??:      ??!       B      ??!       J	>yX?5M@>yX?5M@!>yX?5M@R      ??!       Z	>yX?5M@>yX?5M@!>yX?5M@JCPU_ONLYY?? ],R@b Y      Y@q????S@"?	
host?Your program is HIGHLY input-bound because 72.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nomoderate"s3.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.:
Refer to the TF2 Profiler FAQb?79.6866% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 