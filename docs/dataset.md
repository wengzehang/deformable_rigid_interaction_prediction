# Dataset 

The purpose of this dataset is the learn action effects in scenes with rich interactions between a deformable bag and multiple rigid objects.

This dataset contains 20 different tasks for four actions:
* Pushing an object towards the bag
* Handle motion along a circular trajectory
* Opening the bag
* Lifting the bag

The complete dataset can be downloaded here: INSERT LINK

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


 
