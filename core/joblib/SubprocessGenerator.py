import multiprocessing
import queue as Queue
import threading
import time


class SubprocessGenerator(object):
    
    @staticmethod
    def launch_thread(generator): 
        generator._start()
        
    @staticmethod
    def start_in_parallel( generator_list ):
        """
        Start list of generators in parallel
        """
        for generator in generator_list:
            thread = threading.Thread(target=SubprocessGenerator.launch_thread, args=(generator,) )
            thread.daemon = True
            thread.start()

        while not all ([generator._is_started() for generator in generator_list]):
            time.sleep(0.005)
    
    def __init__(self, generator_func, user_param=None, prefetch=2, start_now=True):
        super().__init__()
        self.prefetch = prefetch
        self.generator_func = generator_func
        self.user_param = user_param
        self.sc_queue = multiprocessing.Queue()
        self.cs_queue = multiprocessing.Queue()
        self.p = None
        self.closed = False
        if start_now:
            self._start()

    def _start(self):
        if self.p == None and not self.closed:
            user_param = self.user_param
            self.user_param = None
            p = multiprocessing.Process(target=self.process_func, args=(user_param,) )
            p.daemon = True
            p.start()
            self.p = p
            
    def _is_started(self):
        return self.p is not None
        
    def process_func(self, user_param):
        self.generator_func = self.generator_func(user_param)
        while True:
            while self.prefetch > -1:
                try:
                    gen_data = next (self.generator_func)
                except StopIteration:
                    self.cs_queue.put (None)
                    return
                self.cs_queue.put (gen_data)
                self.prefetch -= 1
            signal = self.sc_queue.get()
            if signal is None:
                # shutdown signal from the host: exit on our own
                # instead of being terminated mid-pipe-write (a worker
                # killed while the host still has bytes buffered for
                # it can wedge the host's queue feeder thread on a
                # write to a dead reader and hang the host's
                # interpreter shutdown)
                return
            self.prefetch += 1

    def close(self):
        # Deterministic shutdown of the worker process this generator
        # owns (the official DFL lifecycle had no shutdown contract:
        # the daemon worker was only reaped if the host interpreter
        # exited cleanly, and any crashed / killed / hung host left
        # the worker orphaned — spinning and pinning its sample
        # files). Idempotent; a closed generator must not be (re)
        # started.
        if self.closed:
            return
        self.closed = True
        p = self.p
        self.p = None
        if p is not None and p.is_alive():
            # Ask the worker to exit on its own (its main loop polls
            # the sc_queue for host acknowledgements, and a None
            # there is the shutdown signal); only fall back to
            # terminate() if it does not exit in time. The worker
            # must die before any of the pipes it reads are closed,
            # otherwise the queue feeder threads of this process can
            # block forever on a write to a reader that no longer
            # exists and hang the interpreter shutdown.
            try:
                self.sc_queue.put(None)
            except Exception:
                pass
            deadline = time.time() + 5
            while p.is_alive() and time.time() < deadline:
                time.sleep(0.01)
            if p.is_alive():
                p.terminate()
                p.join(10)
        for q in (self.cs_queue, self.sc_queue):
            try:
                # Never let the interpreter shutdown join this
                # queue's feeder thread: if it is still blocked in a
                # pipe write to a worker that is already gone, that
                # join would hang the process forever at exit. The
                # feeder is a daemon thread and is abandoned at exit.
                q.cancel_join_thread()
            except Exception:
                pass
            try:
                q.close()
            except Exception:
                pass

    def __iter__(self):
        return self

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['p']
        return self_dict

    def __next__(self):
        if self.closed:
            raise StopIteration()
        self._start()
        gen_data = self.cs_queue.get()
        if gen_data is None:
            self.p.terminate()
            self.p.join()
            raise StopIteration()
        self.sc_queue.put (1)
        return gen_data
