from matplotlib import pyplot as plt
import numpy as np


class PackCoolingGraph:
    """A visualization of the Pack Cooling simulation using matplotlib"""

    def __init__(self, title=None):
        # self.df = df
        # self.net_worths = np.zeros(len(df['Date']))
        self.render_count = 0

        # Create a figure on screen and set the title
        self.fig = plt.figure(figsize=(5,10))
        self.fig.suptitle(title)

        # Create top subplot for sigma line plot
        self.action_ax = plt.subplot2grid(
            (6, 1), (0, 0), rowspan=2, colspan=1, title="$\sigma(t)$", xlabel="t")

        # Create middle subplot for u meshplot
        self.u_ax = plt.subplot2grid(
            (6, 1), (2, 0), rowspan=2, colspan=1, sharex=self.action_ax, title="u(x,t)", xlabel="x", ylabel="t")
        
        self.u_cb = None

        # Create bottom subplot for w meshplot
        self.w_ax = plt.subplot2grid(
            (6, 1), (4, 0), rowspan=2, colspan=1, sharex=self.action_ax, title="w(x,t)", xlabel="x", ylabel="t")
        
        self.w_cb = None

        # Add padding to make graph easier to view
        plt.subplots_adjust(left=0.2, bottom=0.1,
                            right=0.90, top=0.90, wspace=0.2, hspace=1.0)

        # Show the graph without blocking the rest of the program
        plt.show(block=False)

    def _render_u(self, u):
        # Clear the frame rendered last step
        self.u_ax.clear()
        if self.u_cb:
            self.u_cb.remove()

        x = np.linspace(0,1,u.shape[1])
        y = np.linspace(0,0.1*(u.shape[0]-1),u.shape[0])
        X,Y = np.meshgrid(x,y)

        # Plot u
        im = self.u_ax.pcolormesh(X,Y,u)
        self.u_ax.set_title("u(x,t)")
        self.u_ax.set_xlabel("x")
        self.u_ax.set_ylabel("t")
        self.u_cb = self.fig.colorbar(im,ax=self.u_ax)
        self.u_ax.set_xbound(0.0,1.0)

    def _render_w(self, w):
        # Clear the frame rendered last step
        self.w_ax.clear()
        if self.w_cb:
            self.w_cb.remove()

        x = np.linspace(0,1,w.shape[1])
        y = np.linspace(0,0.1*(w.shape[0]-1),w.shape[0])
        X,Y = np.meshgrid(x,y)

        # Plot w
        im = self.w_ax.pcolormesh(X,Y,w)
        self.w_ax.set_title("w(x,t)")
        self.w_ax.set_xlabel("x")
        self.w_ax.set_ylabel("t")
        self.w_cb = self.fig.colorbar(im,ax=self.w_ax)
        self.w_ax.set_xbound(0.0,1.0)

    def _render_sigma(self, sigma):
        # Clear the frame rendered last step
        self.action_ax.clear()

        t = np.linspace(0,0.1*(sigma.shape[0]-1),sigma.shape[0])

        # Plot sigma
        self.action_ax.plot(t,sigma)
        self.action_ax.set_title("$\sigma(t)$")
        self.action_ax.set_xlabel("t")

    def render(self, u, w, actions):
        self._render_u(u)
        self._render_w(w)
        self._render_sigma(actions)

        plt.savefig("Graph_"+str(self.render_count)+".png")
        self.render_count += 1
        
        # Necessary to view frames before they are unrendered
        plt.pause(0.001)

    def close(self):
        plt.close()