# Deep-Learning-MLB-Pitching

This project utilizes MLB pitch data from 2015-2018 in a few deep-learning models

Model1: Naming the pitch
 - inputs: px, pz, start_speed, end_speed, spin_rate, spin_dir, break_angle, break_length, break_y
 - output classes: changeup, curveball,  eephus, cutter, four-seam fastball, pitchout, splitter, two-seam fastball,
                    intentional ball, knuckle curve, knuckleball, screwball, sinker, slider
 - architecture: transformer encoder classifier
