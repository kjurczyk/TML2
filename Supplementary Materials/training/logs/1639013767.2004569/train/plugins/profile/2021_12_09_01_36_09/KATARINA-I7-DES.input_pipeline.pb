	?????@?????@!?????@	?3??????3?????!?3?????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?????@333333??AGr???@Y????Mb??*	???????@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator??C?l???!nB}?lX@)??C?l???1nB}?lX@:Preprocessing2F
Iterator::Model?<,Ԛ???!?k?y?? @)??_?L??1α1(F???:Preprocessing2P
Iterator::Model::Prefetch?J?4q?!?J???.??)?J?4q?1?J???.??:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?ʡE????!??1??zX@)?~j?t?X?1?jti?g??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?3?????#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	333333??333333??!333333??      ??!       "      ??!       *      ??!       2	Gr???@Gr???@!Gr???@:      ??!       B      ??!       J	????Mb??????Mb??!????Mb??R      ??!       Z	????Mb??????Mb??!????Mb??JCPU_ONLYY?3?????b 