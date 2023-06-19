from bintrees import rbtree

import simpy
import numpy as np
import numpy.random as random
import math

MAXSIMTIME = 10000
VERBOSE = False
LAMBDA = 7
MU = 8.0
POPULATION = 50000000
SERVICE_DISCIPLINE = 'FIFO'
LOGGED = True
PLOTTED = True
# TIMER_INTERRUPT = 0.001
TARGET_LATENCY = 0.02
MIN_GRANULARITY = 0.004


class Job:
    def __init__(self, name, arrtime, duration, vruntime):
        self.name = name
        self.arrtime = arrtime
        self.duration = duration
        self.vruntime = vruntime

    def __str__(self):
        return '%s at %d, length %d' % (self.name, self.arrtime, self.duration)


def SJF(job):
    return job.duration


class Server:
    def __init__(self, env, strat='FIFO'):
        self.env = env
        self.strat = strat
        self.Jobs = rbtree.RBTree()
        self.serversleeping = None
        self.serverRunning = None
        self.target_latency = TARGET_LATENCY
        self.min_granularity = MIN_GRANULARITY
        # self.timer_interrupt = TIMER_INTERRUPT
        ''' statistics '''
        self.waitingTime = 0
        self.idleTime = 0
        self.jobsDone = 0
        self.responseTime = 0
        self.fisrtService = 0
        self.preRunPoint = 0
        self.numOfJobInSys = 0
        self.numOfJobInQ = 0
        self.contextSwitchArr = list()
        ''' register a new server process '''
        env.process(self.serve())

    def serve(self):
        while True:
            ''' do nothing, just change server to idle
              and then yield a wait event which takes infinite time
            '''
            if self.Jobs.count == 0:
                self.serversleeping = env.process(self.waiting(self.env))
                t1 = self.env.now
                '''yield an event "server sleeping". env.process return an event.'''
                yield self.serversleeping
                ''' accumulate the server idle time'''
                self.idleTime += self.env.now - t1
            else:
                if self.Jobs.count > self.target_latency / self.min_granularity:
                    timeslot = self.min_granularity
                else:
                    timeslot = self.target_latency / self.Jobs.count
                ''' get the first job to be served'''
                j = self.Jobs.pop_min()[1]

                # kiem tra co phai job duoc phuc vu lan dau hay khong
                if j.vruntime == 0: # lan dau
                    self.fisrtService += self.env.now - j.arrtime

                # if LOGGED:
                #     qlog.write('%.4f\t%d\t%d\n'
                #                % (self.env.now, 1 if len(self.Jobs) > 0 else 0, len(self.Jobs)))

                if VERBOSE:
                    print(f'Get new job {j.name} at {env.now}, vruntime = {j.vruntime}')

                # thoi gian con lai cua job
                remaining_duration = j.duration - j.vruntime

                # cong don vruntime ?
                if self.Jobs.count > 0:
                    if VERBOSE:
                        print(f'smallest is {self.Jobs.min_item()[1].name}, {self.Jobs.min_item()[1].vruntime}')
                    # tinh toan thoi gian cach biet giua vruntime cua job va vruntime be nhat trong RBtree.
                    # co the dam bao la vruntime be nhat trong RBtree luon > vruntime cua job hien tai.
                    diff_vruntime = self.Jobs.min_item()[1].vruntime - j.vruntime
                    # neu khoang thoi gian con lai cua job < khoang cach toi job be nhat trong RBtree, dieu do co nghia
                    # la, trong qua trinh job chay sao cho vruntime > job be nhat trong RBtree, job da hoan thanh xong
                    # som hon. nguoc lai, neu remaining_duration > diff_vruntime, nghia la sau khi duoi kip job be nhat
                    # trong RBtree, job van chua hoan thanh ma can chay them. luc nay minh se cho job chay roi sau do
                    # add lai vao RBtree, de lay job be nhat ra xu ly tiep.
                    if remaining_duration > diff_vruntime:
                        multi = int(diff_vruntime / timeslot) + 1
                        temp_vruntime = timeslot * multi
                        if temp_vruntime > remaining_duration:
                            temp_vruntime = remaining_duration
                    else:  # remaining_duration <= self.Jobs.min_item()[1].vruntime
                        temp_vruntime = remaining_duration
                else:
                    temp_vruntime = remaining_duration

                if VERBOSE:
                    print(f'temp_vruntime = {temp_vruntime}, with timeslot = {timeslot}')

                ''' run the job'''
                # result_vruntime la 1 list, dua vao ham running_job de khi process chay xong, ket qua se duoc
                # append vao list nay, sau do lay ra de xu ly. day la mot trong nhung phuong phap de minh luu gia tri
                # vao tham so ham de xai sau khi ham ket thuc ma khong can thong qua return.

                result_vruntime = list()
                self.serverRunning = env.process(self.running_job(env, temp_vruntime, timeslot, result_vruntime))
                yield self.serverRunning

                actual_runtime = result_vruntime.pop()
                j.vruntime += actual_runtime

                remaining_duration = j.duration - j.vruntime

                if remaining_duration > 0:
                    if VERBOSE:
                        print(f'Job {j.name} run for {actual_runtime} at {self.env.now},'
                              f' has run for {j.vruntime} out of {j.duration}')
                    # ok, IF, THIS NEW vruntime happend to have the same value of the smallest vruntime in the tree,
                    # we may "cheat" by insert it again, but with j.vruntime - 1. so that this task will run again, and
                    # it will larger than the smallest key.
                    before_insert = self.Jobs.count
                    if self.Jobs.get(j.vruntime) is not None:
                        # re-schedule the job (why? if this job have vruntime value the same as a job in the tree, the
                        # job in the tree will be "replace" by this job. so, we have to insert to the tree another
                        # key value. we may try to add or substract from the job vruntime, and the amount is no longer
                        # than min_granularity.)

                        reduced_vruntime = j.vruntime + (1 / (env.now + 1)) * self.min_granularity / 2
                        self.Jobs.insert(reduced_vruntime, j)
                    else:
                        # re-schedule the job (normally)
                        self.Jobs.insert(j.vruntime, j)
                    # perform a context-switch if next job is different:
                    after_insert = self.Jobs.count
                    if after_insert - before_insert == 0:
                        raise Exception("Jobs insertion gone wrong")

                    if self.Jobs.min_item()[1].name != j.name:
                        if VERBOSE:
                            print("Context-switch will happen")
                            # context-switch: save the state of the current job, and load a new one
                        con_sw = self.env.process(self.context_switch(self.env, j))
                        yield con_sw
                elif remaining_duration == 0 or abs(0 - remaining_duration) < 10 ** (-15):
                    # remaining_duration == 0, nhung vi co 1 chut sai so do kieu du lieu FLOAT,
                    # minh buoc phai so sanh 1 so rat nho gan so 0
                    self.waitingTime += self.env.now - j.arrtime - j.vruntime
                    self.jobsDone += 1
                    if VERBOSE:
                        print(f'Job {j.name} done at {self.env.now}, has run for {j.vruntime}')
                    # tinh response time
                    self.responseTime += self.env.now - j.arrtime
                    # tinh khoang cach thoi gian
                    tempTime = self.env.now - self.preRunPoint
                    if tempTime < 0:
                        raise Exception("tempTime < 0,",tempTime)
                    # tong job trong he thong trong khoang thoi gian tempTime
                    self.numOfJobInSys += (len(self.Jobs) + 1) * tempTime
                    self.numOfJobInQ += len(self.Jobs) * tempTime
                    # cap nhat preRunPoint
                    self.preRunPoint = self.env.now
                else:
                    raise Exception(
                        f"Why the fuck remaining duration is < than 0 ??? at {env.now}, {j.name}, {j.vruntime} + "
                        f"{actual_runtime} = {j.duration}")

    def context_switch(self, enviroment, job):
        yield self.env.timeout(job.duration / 100000)
        self.contextSwitchArr.append(job.duration / 100000)
        if VERBOSE:
            print(f"Context-switch completed at {enviroment.now}")

    def waiting(self, enviroment):
        try:
            if VERBOSE:
                print('Server is idle at %f' % enviroment.now)
            yield self.env.timeout(MAXSIMTIME)
        except simpy.Interrupt as i:
            if VERBOSE:
                print('Server waken up and works at %f' % enviroment.now)

    def running_job(self, environment, temp_vruntime, timeslot, result_vruntime):
        # RUN THE JOB
        start = environment.now
        try:
            yield environment.timeout(temp_vruntime)
            result_vruntime.append(temp_vruntime)
            if VERBOSE:
                print(f'Job run normally for {temp_vruntime} by timeslot {timeslot} at {environment.now}')
        except simpy.Interrupt as i:
            if VERBOSE:
                print(f"{i} at {environment.now}")
            running_time = environment.now - start
            mul = int(running_time / timeslot) + 1
            interrupted_runtime = timeslot * mul
            if interrupted_runtime > temp_vruntime:
                interrupted_runtime = temp_vruntime
            remaining_runtime = interrupted_runtime - running_time

            complete = False
            start_remaining = environment.now
            '''try to run the job to complete the current time slot.'''
            while not complete:
                try:
                    yield environment.timeout(remaining_runtime - (environment.now - start_remaining))
                    complete = True
                    if VERBOSE:
                        print("Too many jobs! but at least, I'm TRUE")
                except:
                    if VERBOSE:
                        print(f'Interrupted mid_point at {environment.now}')
                    # remaining_runtime = remaining_runtime - (environment.now - start_remaining)

            result_vruntime.append(interrupted_runtime)
            if VERBOSE:
                print(
                    f"Interrupted. Tried to run job for {interrupted_runtime} out off {temp_vruntime} by "
                    f"timeslot {timeslot} at {environment.now}")


class JobGenerator:
    def __init__(self, env, server, nrjobs=10000000, lam=5.0, mu=8.0):
        self.server = server
        self.nrjobs = nrjobs
        self.interarrivaltime = 1 / lam
        self.servicetime = 1 / mu
        env.process(self.generatejobs(env))

    def generatejobs(self, env):
        i = 1
        while True:
            '''yield an event for new job arrival'''
            job_interarrival = random.exponential(self.interarrivaltime)
            yield env.timeout(job_interarrival)
            if VERBOSE:
                print(f"a task has just come at {env.now}")

            tempTime = env.now - self.server.preRunPoint
            if tempTime < 0:
                raise Exception("tempTime < 0,",tempTime)
            if self.server.serverRunning is not None and not self.server.serverRunning.triggered:
                self.server.numOfJobInSys += tempTime * (len(self.server.Jobs) + 1)
            else:
                self.server.numOfJobInSys += tempTime * len(self.server.Jobs)

            self.server.numOfJobInQ += tempTime * len(self.server.Jobs)
            self.server.preRunPoint = env.now

            ''' generate service time and add job to the list'''
            job_duration = random.exponential(self.servicetime)
            init_key = 0

            if self.server.Jobs.get(init_key) is not None:
                init_key = init_key + (1 / (env.now + 1)) * self.server.min_granularity / 2

            before_insert = self.server.Jobs.count

            self.server.Jobs.insert(init_key, Job('Job %s' % i, env.now, job_duration, 0))

            after_insert = self.server.Jobs.count
            if after_insert - before_insert == 0:
                raise Exception("Insert Init Job gone wrong.")

            if True:
                print('job %d: t = %f, l = %f, dt = %f'
                      % (i, env.now, job_duration, job_interarrival))
            i += 1

            ''' if server is idle, wake it up'''
            if not self.server.serversleeping.triggered:
                self.server.serversleeping.interrupt('Wake up, please.')
            ''' if server is serving a job and a job suddenly come, interrupt server.'''
            if self.server.serverRunning is not None and not self.server.serverRunning.triggered:
                self.server.serverRunning.interrupt("New job comming!")


# if LOGGED:
#     qlog = open('mm1-l%d-m%d.csv' % (LAMBDA, MU), 'w')
#     qlog.write('0\t0\t0\n')

env = simpy.Environment()
MyServer = Server(env, SERVICE_DISCIPLINE)
MyJobGenerator = JobGenerator(env, MyServer, POPULATION, LAMBDA, MU)
env.run(MAXSIMTIME)
#
# env.run(until=MAXSIMTIME)
#
# # if LOGGED:
# #     qlog.close()
#
RHO = LAMBDA / MU
print('Arrivals               : %d' % (MyServer.jobsDone))
print('Utilization            : %.2f/%.2f'
      % (1.0 - MyServer.idleTime / MAXSIMTIME, RHO))
print('Mean waiting time      : %.2f/%.2f'
      % (MyServer.waitingTime / MyServer.jobsDone, RHO ** 2 / ((1 - RHO) * LAMBDA)))
print( 'Mean response time     : %.2f/%.2f'
    % (MyServer.responseTime/MyServer.jobsDone, (1/MU)/(1 - RHO)))
# print( 'Mean number of jobs    : %.2f/%.2f'
#     % (LAMBDA * MyServer.responseTime / MyServer.jobsDone, RHO/(1 - RHO)))
# print( 'Mean number of jobs in queue : %.2f/%.2f'
#     % (LAMBDA * (MyServer.waitingTime/MyServer.jobsDone), RHO**2 / (1 - RHO)))
print( 'Mean number of jobs   : %.2f/%.2f'
    % (MyServer.numOfJobInSys / MAXSIMTIME, RHO/(1 - RHO)))
print( 'Mean number of jobs in queue : %.2f/%.2f'
    % (MyServer.numOfJobInQ / MAXSIMTIME, RHO**2 / (1 - RHO)))
print( 'Mean time from arrival to first service : %f'
    % (MyServer.fisrtService / MyServer.jobsDone))
print( 'Mean context switch time : %f'
    % (sum(MyServer.contextSwitchArr) / len(MyServer.contextSwitchArr)))
print( 'Rate of number of jobsDone divided by total number of jobs : %f'
    % (MyServer.jobsDone / (len(MyServer.Jobs) + MyServer.jobsDone)))
print( 'Number of remain jobs :', len(MyServer.Jobs))




#
# if LOGGED and PLOTTED:
#     import matplotlib.pyplot as plt
#
#     log = np.loadtxt('mm1-l%d-m%d.csv' % (LAMBDA, MU), delimiter='\t')
#     plt.subplot(2, 1, 1)
#     plt.xlabel('Time')
#     plt.ylabel('Queue length')
#     plt.step(log[:200, 0], log[:200, 2], where='post')
#     plt.subplot(2, 1, 2)
#     plt.xlabel('Time')
#     plt.ylabel('Server state')
#     plt.yticks([0, 1], ['idle', 'busy'])
#     # plt.step( log[:200,0], log[:200,1], where='post' )
#     plt.fill_between(log[:200, 0], 0, log[:200, 1], step="post", alpha=.4)
#     plt.tight_layout()
#     plt.show()
