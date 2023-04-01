# Dataset 

The purpose of this dataset is the learn action effects in scenes with rich interactions between a deformable bag and multiple rigid objects.

This dataset contains 20 different tasks for four actions:
* Pushing an object towards the bag
* Handle motion along a circular trajectory
* Opening the bag
* Lifting the bag

For each task, we simulate 1,000 6-sec trajectories, and record the scene state 10 times per second, 
which results in 60,000 recorded time steps.
The simulated task data is split into training (80\%), validation (10\%), and test set (10\%).
We vary actions and task parameters to create data for 20 different tasks.

The complete dataset can be downloaded here: https://kth-my.sharepoint.com/:f:/g/personal/zehang_ug_kth_se/EvUoLXferkVJj2Lgilcoh24Bm3nZn8KgzN4zeBQ3cBxEiA?e=3Gjbhu



## Tasks

The following parameters differentiate the tasks:
* Bag Stiffness: Is the bag's material stiff or soft?
* Bag Content: Is the bag empty or is a rigid object inside?
* Left/Right Handle State: Is the handle fixed in place, released or moved along a trajectory?
* Controlled Object: Which object is actively manipulated during the action?
* Action: Which action is executed?

| Task id | Bag Stiffness | Bag Content   | Left Handle State | Right Handle State | Controlled Object | Action                 |
|---------|---------------|---------------|-------------------|--------------------|-------------------|------------------------|
| 1/11    | Soft/Stiff    | Object Inside | Fixed             | Fixed              | Sphere            | Pushing an Object      |
| 2/12    | Soft/Stiff    | Empty         | Fixed             | Fixed              | Sphere            | Pushing an Object      |
| 3/13    | Soft/Stiff    | Object Inside | Moving            | Fixed              | Left Hand         | Circular Handle Motion |
| 4/14    | Soft/Stiff    | Empty         | Moving            | Fixed              | Left Hand         | Circular Handle Motion |
| 5/15    | Soft/Stiff    | Object Inside | Moving            | Released           | Left Hand         | Circular Handle Motion |
| 6/16    | Soft/Stiff    | Empty         | Moving            | Released           | Left Hand         | Circular Handle Motion |
| 7/17    | Soft/Stiff    | Object Inside | Moving            | Fixed              | Left Hand         | Opening the Bag        |
| 8/18    | Soft/Stiff    | Empty         | Moving            | Fixed              | Left Hand         | Opening the Bag        |
| 9/19    | Soft/Stiff    | Object Inside | Moving            | Released           | Left Hand         | Lifting the Bag        |
| 10/20   | Soft/Stiff    | Empty         | Moving            | Released           | Left Hand         | Lifting the Bag        |


 
## H5 dataset key for each task
| H5 key name        | Description                                                                                                                                    | Shape              |
|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|
| randomname         | Random name for every trajectory.                                                                                                              | (100,)             |
| clothid            | Id of the bag object. We provide one bag with a fixed index "20".                                                                              | (100,)             |
| clothmaterial_bend | Unity Obi bending parameter for the bag material. 0.1 for soft, 0.01 for stiff.                                                                | (100,)             |
| clothmaterial_dist | Unity Obi distance parameter for the bag material.                                                                                             | (100,)             |
| posCloth           | 3D Position of bag mesh vertices.                                                                                                              | (100, 61, 1277, 3) |
| veloCloth          | Velocity of bag vertices in Unity.                                                                                                             | (100, 61, 1277, 3) |
| numRigid           | Number of rigid objects in the scene, without counting the sphere effector. Use it to filter the position of rigid objects.                    | (100,)             |
| posRigid           | Position and radius of the free sphere. Only the first "numRigid" pose are valid.                                                              | (100, 61, 10, 4)   |
| veloRigid          | Velocity of rigid sphere in Unity.                                                                                                             | (100, 61, 10, 3)   |
| ballinside         | Number of sphere inside the bag.                                                                                                               | (100,)             |
| posEffector        | Position and radius of the sphere effector. Only valid in the pushing action task.                                                             | (100, 61, 1, 4)    |
| graspnum_l         | Number of grasped points for the first handle. Only valid when the left handle is not released.                                                | (100, 61)          |
| graspnum_r         | Number of grasped points for the first handle. Only valid when the right handle is not released.                                               | (100, 61)          |
| graspind_l         | The indices of grasped vertices of the left handle. Only valid if left handle is not released. Only first "graspnum_l" indices are avaliable.  | (100, 61, 20)      |
| graspind_r         | The indices of grasped vertices of the right handle. Only valid if right handle is not released.Only first "graspnum_r" indices are avaliable. | (100, 61, 20)      |
| circletype         | Circular type of the motion. We drawing circle in different coordinate planes.                                                                 | (100,)             |
| initSpeedEffector  | Speed of effetcor. Only valid in the pushing action task.                                                                                      | (100,)             |
| initPosEffector    | The position and radius of creating sphere effector.                                                                                           | (100, 1, 4)        |
| sampleflag         | The way of creating the sphere effector. Only valid in the pushing action task.                                  | (100,)             |
| towardsflag        | The target of effector pushing. Only valid in the pushing action task.                                     | (100,)             |
| initActEffector    | Deprecated key.                                                                                                                                | (100, 1, 3)        |
| gripperspeed       | Deprecated key.                                                                                                                                | (100,)             |
| posWall            | Deprecated key.                                                                                                                                | (100, 61, 5, 9)    |
| circlemode         | Deprecated key.                                                                | (100,)             |
| circleclock        | Deprecated key                                                           | (100,)             |

## Label Mapping

The circle task parameter is only avaliable when the left handle is with circular movement.

| circletype     | Description                                                                                                                                    |
|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| 0         | Draw the circle in x-z coordinate plane                                                                |
| 1         | Draw the circle in x-y coordinate plane                                                                |
| 2         | Draw the circle in z-y coordinate plane                                                                |                                                            |

| sampleflag     | Description                                                                                                                                    |
|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| 0         | The sphere effector is created near a rigid sphere                                                                |
| 1         | The sphere effector is created near the deformable bag                                                                |

| towardsflag     | Description                                                                                                                                    |
|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| 0         | The sphere effector is moved towards a rigid sphere                                                                |
| 1         | The sphere effector is moved towards the deformable bag                                                                |

