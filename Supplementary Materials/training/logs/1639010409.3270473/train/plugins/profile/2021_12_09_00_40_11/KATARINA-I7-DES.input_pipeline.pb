	)?Ǻ?
@)?Ǻ?
@!)?Ǻ?
@	???C?J?????C?J??!???C?J??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$)?Ǻ?
@^?I+??A?g??s?	@Y]m???{??*	????̤{@2g
0Iterator::Model::Prefetch::FlatMap[0]::GeneratorOjM???!~c?h=?W@)OjM???1~c?h=?W@:Preprocessing2F
Iterator::ModeljM????!?`c??E@)?{??Pk??1?j?#U@:Preprocessing2P
Iterator::Model::Prefetcha??+ey?!UT???m??)a??+ey?1UT???m??:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMaph??s???!??)b??W@)?J?4a?1Arf:?b??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???C?J??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	^?I+??^?I+??!^?I+??      ??!       "      ??!       *      ??!       2	?g??s?	@?g??s?	@!?g??s?	@:      ??!       B      ??!       J	]m???{??]m???{??!]m???{??R      ??!       Z	]m???{??]m???{??!]m???{??JCPU_ONLYY???C?J??b 