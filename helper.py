import matplotlib.pyplot as plt
from IPython import display
import time

plt.ion()

def plot(score, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training........')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(score)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    # plt.show()
    plt.text(len(score)-1,score[-1],str(score[-1]))
    plt.text(len(mean_scores)-1,mean_scores[-1],str(mean_scores[-1]))

# for x in range(0,20):
#     plot(x,x+x)
#     time.sleep(2)