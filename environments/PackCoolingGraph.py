from matplotlib import pyplot as plt
import numpy as np


class PackCoolingGraph:
    """A stock trading visualization using matplotlib made to render OpenAI gym environments"""

    def __init__(self, title=None):
        # self.df = df
        # self.net_worths = np.zeros(len(df['Date']))

        # Create a figure on screen and set the title
        fig = plt.figure(figsize=(10,10))
        fig.suptitle(title)

        # Create top subplot for sigma line plot
        self.action_ax = plt.subplot2grid(
            (6, 2), (0, 0), rowspan=2, colspan=2, title="$\sigma(t)$", xlabel="t")

        # Create middle subplot for u meshplot
        self.u_ax = plt.subplot2grid(
            (6, 2), (2, 0), rowspan=8, colspan=1, sharex=self.action_ax, title="u(x,t)", xlabel="x", ylabel="t")

        # Create bottom subplot for w meshplot
        self.w_ax = plt.subplot2grid(
            (6, 2), (2, 1), rowspan=8, colspan=1, sharex=self.action_ax, title="w(x,t)", xlabel="x", ylabel="t")
        
        # Add padding to make graph easier to view
        plt.subplots_adjust(left=0.1, bottom=0.1,
                            right=0.90, top=0.90, wspace=0.2, hspace=1.0)

        # Show the graph without blocking the rest of the program
        plt.show(block=False)

    def _render_u(self, u):
        # Clear the frame rendered last step
        self.u_ax.clear()

        x = np.linspace(0,1,u.shape[1])
        y = np.linspace(0,0.1*(u.shape[0]-1),u.shape[0])
        X,Y = np.meshgrid(x,y)

        # Plot u
        self.u_ax.pcolormesh([X,Y],u)

    def _render_w(self, w):
        # Clear the frame rendered last step
        self.w_ax.clear()

        x = np.linspace(0,1,w.shape[1])
        y = np.linspace(0,0.1*(w.shape[0]-1),w.shape[0])
        X,Y = np.meshgrid(x,y)

        # Plot w
        self.w_ax.pcolormesh([X,Y],w)

    def _render_sigma(self, sigma):
        # Clear the frame rendered last step
        self.action_ax.clear()

        t = np.linspace(0,0.1*(sigma.shape[0]-1),sigma.shape[0])

        # Plot sigma
        self.action_ax.plot(t,sigma)

    def render(self, u, w, actions):
        self._render_u(u)
        self._render_w(w)
        self._render_sigma(actions)
        
        # Necessary to view frames before they are unrendered
        plt.pause(0.001)

    def close(self):
        plt.close()