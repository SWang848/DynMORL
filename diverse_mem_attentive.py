import numpy as np

class ReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity
        self.data = np.repeat(((None, None, None), ), capacity, axis=0)
        self.write = 0 
    
    def add(self, data, write=None):
        write = write if write is not None else self.write

        replaced_data = np.copy(self.data[write])
        self.data[write] = data
        return replaced_data

    def get(self, s):
        return self.data[s]


class DiverseMemory:


    def __init__(
            self,
            main_capacity,
            sec_capacity,
            trace_diversity=True,
            crowding_distance=True,
            value_function=lambda trace, trace_id, memory_indices: np.random.random(1)):
        
        self.len = 0
        self.trace_diversity = trace_diversity
        self.value_function = value_function
        self.crowding_distance = crowding_distance
        self.capacity = main_capacity + sec_capacity
        self.main_capacity = main_capacity
        self.sec_capacity = sec_capacity
        self.secondary_traces = []

        self.data = np.repeat(((None, None, None), ), self.capacity, axis=0) 
        # data (transition) transition (trace_id, sample, pred_idx) sample (state, action, reward, next_state, terminal, extra)
        self.write = 0

    def main_mem_is_full(self):

        return self.data[self.write][0] is not None

    def add(self, sample, trace_id=None, pred_idx=None):
        self.len = min(self.len + 1, self.capacity)

        transition = (trace_id, sample, None if pred_idx is None else pred_idx)

        if self.main_mem_is_full():
            end = self.extract_trace(self.write)
            self.move_to_sec(self.write, end)

        _ = self.add_sample(transition, self.write)
        self.write = (self.write + 1) % self.main_capacity

        return self.write-1
    
    def add_sample(self, transition, write=None):
        replaced_data = np.copy(self.data[write])

        self.data[write] = transition

        return replaced_data

    def extract_trace(self, start):
        trace_id = self.data[start][0]

        end = (start + 1) % self.main_capacity

        if not self.trace_diversity:
            return end
        if trace_id is not None:
            while self.data[end][0] == trace_id
                end = (end + 1) % self.main_capacity
                if end == start:
                    break
        
        return end
    
    def move_to_sec(self, start, end):

        if end <= start:
            indices = np.r_[start:self.main_capacity, 0:end]
        else:
            indices = np.r_[start:end]
        
        if not self.trace_diversity:
            assert len(indices) == 1
        
        trace = np.copy(self.data[indices])

        write_indices = self.get_sec_write(self.secondary_trace, trace)

        if write_indices is not None and len(write_indices) >= len(trace):
            for i, (w, t) in enumerate(zip(write_indices, trace)):

                self.data[w] = t

                if i > 0:
                    self.data[w][2] = write_indices[i - 1]
            if not self.trace_diversity:
                assert len(trace) == 1
            self.secondary_traces.append((trace, write_indices))
        
        self.remove_trace((None, indices))
    
    def remove_trace(self, trace):
        _, trace_idx = trace
        for i in trace_idx:
            self.data[i] = (None, None, None) 

    def get_sec_write(self, secondary_traces, trace, reserved_idx=None): 
        if reserved_idx is None:
            reserved_idx = []

        if len(trace) > self.sec_capacity:
            return None
                
        if len(reserved_idx) >= len(trace):
            return reserved_idx[:len(trace)]
        
        free_spots = [i + self.main_capacity for i in range(self.sec_capacity) if (self.data[self.main_capacity+i][1]) is None]

        if len(free_spots) > len(reserved_idx):
            return self.get_sec_write(secondary_traces, trace, free_spots[:len(trace)])
        
        idx_dist, _ = self.sec_distances(secondary_traces)

        i, _ = min(idx_dist, key=lambda d: d[1])

        _, trace_idx = secondary_traces[i]
        reserved_idx += trace_idx

        self.remove_trace(secondary_traces[i])

        del secondary_traces[i]
        return self.get_sec_write(secondary_traces, trace, reserved_idx)
    
    def sec_distance(self, traces):
        values = [self.get_trace_value(tr) for tr in traces]
        if self.crowding_distance:
            distances = self.crowd_dist(values)
        else:
            distances = values
        
        return [(i, d) for i, d in enumerate(distances)], values
    
    def crowd_dist(self, datas):
        points = np.array([Object() for _ in datas])
        dimensions = len(datas[0])
        for i, d in enumerate(datas):
            points[i].data = d
            points[i].i = i
            points[i].distance = 0.

        # Compute the distance between neighbors for each dimension and add it to
        # each point's global distance
        for d in range(dimensions):
            points = sorted(points, key=lambda p: p.data[d])
            spread = points[-1].data[d] - points[0].data[d]
            for i, p in enumerate(points):
                if i == 0 or i == len(points) - 1:
                    p.distance += INF
                else:
                    p.distance += (
                        points[i + 1].data[d] - points[i - 1].data[d]) / spread

        # Sort points back to their original order
        points = sorted(points, key=lambda p: p.i)
        distances = np.array([p.distance for p in points])

        return distances
    
    def 