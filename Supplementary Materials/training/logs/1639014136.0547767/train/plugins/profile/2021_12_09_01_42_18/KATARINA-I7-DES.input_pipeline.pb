	x$(?@x$(?@!x$(?@	of?X+?R@of?X+?R@!of?X+?R@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$x$(?@?d?`TR??A?B?i?q??Y,Ԛ?g@*	   ??@2P
Iterator::Model::Prefetch??|?5@!?o?%WWI@)??|?5@1?o?%WWI@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap????@!V?l?IH@)M?O/	@1????Q?C@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?X????!q??+A?!@)?X????1q??+A?!@:Preprocessing2F
Iterator::Model|??PkZ@!?<???I@)%u???1H󍛪??:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor????MbP?!?)W"g?)????MbP?1?)W"g?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 75.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2s3.4 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9of?X+?R@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?d?`TR???d?`TR??!?d?`TR??      ??!       "      ??!       *      ??!       2	?B?i?q???B?i?q??!?B?i?q??:      ??!       B      ??!       J	,Ԛ?g@,Ԛ?g@!,Ԛ?g@R      ??!       Z	,Ԛ?g@,Ԛ?g@!,Ԛ?g@JCPU_ONLYYof?X+?R@b 