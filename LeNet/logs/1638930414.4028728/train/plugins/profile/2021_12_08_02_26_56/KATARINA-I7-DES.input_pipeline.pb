	?/?$F @?/?$F @!?/?$F @	??N?????N???!??N???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?/?$F @?:M???ANbX9?@Y ?o_ι?*	gffff??@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?#??????!??j?ϸW@)?#??????1??j?ϸW@:Preprocessing2F
Iterator::Model??~j?t??!3??
-?@)?D???J??1\??'?@:Preprocessing2P
Iterator::Model::Prefetch?]K?=??!?????)?]K?=??1?????:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap????9#??!m?Q/}?W@)?+e?Xw?1?D{s?V??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??N???>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?:M????:M???!?:M???      ??!       "      ??!       *      ??!       2	NbX9?@NbX9?@!NbX9?@:      ??!       B      ??!       J	 ?o_ι? ?o_ι?! ?o_ι?R      ??!       Z	 ?o_ι? ?o_ι?! ?o_ι?JCPU_ONLYY??N???b 