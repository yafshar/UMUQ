from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# set Latex environment
plt.rc('text', usetex=True)
plt.rc('font', family='serif')



def plot_target_scatter(index_test, data_frame, cut_off, case_name):


    # visualization with cut off
    data_frame = data_frame[~(data_frame['funcValue'] <= cut_off)]

    # color for visualization
    color_funV = data_frame['funcValue'].astype(int)
    color_funV = color_funV.apply(lambda x: x)

    # plot
    fig = plt.figure(figsize=(20, 16))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('2f')
    ax.set_ylabel('3f')
    ax.set_zlabel('4f')
    ax.view_init(16, 48)
    ax.set_xlim([3.3, 3.5])
    ax.set_ylim([6.1, 7.5])
    ax.set_zlim([0.14, 0.36])
    ax.set_axis_bgcolor('white')

    sca_plot = ax.scatter(data_frame['2f'], data_frame['3f'], data_frame['4f'],
                          s=20, c=color_funV, cmap='hot', depthshade=0, alpha=1)
    cbar = plt.colorbar(sca_plot, fraction=0.046, pad=0.15, shrink=0.8, format='%.2f')
    cbar.set_clim(cut_off, 30.0)
    cbar.ax.tick_params(labelsize=30)

    tot_num_data = data_frame.shape[0]

    fig.set_dpi(100)
    # fig.set_size_inches(18.5, 10.5, forward=True)
    plt.title('Accumulated No. ' + str(index_test + 1) + ' MCMC data with log(prob) cut off value: ' + str(cut_off),
              fontsize=30)

    ax.text(3, 6, 0.2, r'$N_{total}$ = ' + str(tot_num_data), fontsize=20)

    plt.savefig(case_name +'_no_data_trained_' + str(index_test + 1) + '_cut_off' + str(cut_off) + '.png', bbox_inches='tight')
    plt.close()