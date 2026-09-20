from pathlib import Path

'''
You can implement your own SampleGenerator
'''
class SampleGeneratorBase(object):


    def __init__ (self, debug=False, batch_size=1):
        self.debug = debug
        self.batch_size = 1 if self.debug else batch_size
        self.last_generation = None
        self.active = True

    def set_active(self, is_active):
        self.active = is_active

    def close(self):
        # Deterministic shutdown of everything this sample generator
        # owns. Order matters: first the index-host helper threads
        # (they keep writing index results into queues whose readers
        # are the worker processes — they must stop while those
        # workers are still alive and draining, otherwise this
        # process can be left with queued writes to pipes whose
        # reader is gone, and the interpreter shutdown hangs on them).
        # Then the generators themselves (subprocess generators ask
        # their worker process to exit, falling back to terminate +
        # join; in-process ones have a no-op close). Called by the
        # model teardown (ModelBase.finalize) so a session can never
        # leave owned worker processes behind.
        for host in (getattr(self, 'index_host', None),
                     getattr(self, 'ct_index_host', None)):
            if host is not None:
                try:
                    host.stop()
                except Exception:
                    pass
        for generator in (getattr(self, 'generators', None) or []):
            generator.close()
        self.active = False

    def generate_next(self):
        if not self.active and self.last_generation is not None:
            return self.last_generation
        self.last_generation = next(self)
        return self.last_generation

    #overridable
    def __iter__(self):
        #implement your own iterator
        return self

    def __next__(self):
        #implement your own iterator
        return None
    
    #overridable
    def is_initialized(self):
        return True