# Pipeline runtime

The pipeline layer owns typed topology and request movement. It does not implement
model tensor math.

Initial components will include stage lifecycle, immutable payload/stream
messages, bounded queues, serial scheduling, cancellation propagation, fan-in,
fan-out, and metrics hooks. A stage is a scheduling/ownership boundary and does
not imply a dedicated thread.
