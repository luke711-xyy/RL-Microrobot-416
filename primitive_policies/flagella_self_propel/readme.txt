The codes to achieve a chain swimmer's propulsion will generate three directories (traj, traj2, trajp) to store the swimmer's data and one directory for saving policies. 

In the "traj" directory, all files should be concatenated sequentially according to their indexes in the file names, forming a complete dataset of the swimmer's information.
Each line corresponds to: X coordinate of centroid, Y coordinate of centroid, first link's global angle, angle at hinges

In the "traj2" directory, all files should be concatenated sequentially according to their indexes in the file names, forming a complete dataset of the swimmer's information.
Each line corresponds to: X coordinate of centroid, Y coordinate of centroid, X coordinate of the swimmer's back end, Y coordinate of the swimmer's back end.

In the "trajp" directory, all files should be concatenated sequentially according to their indexes in the file names, forming a complete dataset of the pressure.
Each line corresponds to: pressure at the back end, pressure at all hinges, pressure at the front end

In the "policy" directory, each folder corresponds to a policy model saved by RLlib in the training. The policy with a larger number in the title represents the more well-trained one.
