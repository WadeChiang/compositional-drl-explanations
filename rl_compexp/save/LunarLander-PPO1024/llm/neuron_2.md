 Key Trigger Conditions:

1. **Vy Upward or Y Coord Near Ground** (Weight: 10.0)
   - When the vertical velocity (\( V_y \)) is upward (positive) or the lander is close to the ground.

2. **Combination of Vy Low/Upward or Y Coord Near Ground, Large Angle, Angular Velocity High** (Weight: 9.5)
   - When vertical velocity is low or upward, close to the ground, exhibits a large angle, or has a high angular velocity.

3. **Angle Near 0 & Y Coord Far From Ground without Right Leg Contact or X Coord Outside Wider Center** (Weight: 8.5)
   - When the lander is nearly horizontal while far from the ground and the right leg is not in contact, or the lander is significantly deviating from the center line.

4. **Not Right Leg Contact with Vy Low/Upward or X Coord Outside Wider Center** (Weight: 8.4)
   - When the right leg is not touching the ground and \( V_y \) is low or upward, or far from the center line.

5. **Vy Low or Vy Upward** (Weight: 7.2)
   - When the vertical velocity is either low or upward.

 Practical Decision Guide:

To guide the effective deployment of the main engine, operators can use the following checklist:

1. **Check Vertical Velocity \( V_y \) and Altitude (Y Coord)**:
   - If \( V_y \) is upward or the lander is near the ground, consider firing the main engine to stabilize or slow the ascent.
   - When \( V_y \) is low and the lander is near the ground or exhibits a large angle, firing the main engine might be necessary to correct the descent path.

2. **Monitor Angular Stability and High Angular Velocity**:
   - If the lander has a large angle or high angular velocity, firing the main engine may be crucial to stabilize the orientation.

3. **Evaluate Horizontal Position (X Coord) and Leg Contact**:
   - If the lander is significantly outside the wider center range or if the right leg is not in contact with the ground, firing the main engine can help regain a more stable position.

4. **Leg Contacts**:
   - If the right leg is not in contact with the ground and vertical velocity is low or upward, the main engine might need to be fired to adjust the lander's orientation and trajectory.

 Comprehensive Strategy:

- **Near Ground Stabilization:**
  - **Condition:** Y Coord Near Ground
  - **Action:** If \( V_y \) is upward or low and the lander is near the ground while exhibiting a high angle or angular velocity, fire the main engine to decelerate and stabilize.

- **Far from Ground Adjustment:**
  - **Condition:** Y Coord Far From Ground, \( V_y \) Manageable
  - **Action:** If the lander is far from the ground but has a low \( V_y \), and shows signs of instability (large angle or high angular velocity), fire the main engine to control the descent and maintain a near-horizontal orientation.

- **Angular Corrections:**
  - **Condition:** Large Angle or High Angular Velocity
  - **Action:** Fire the main engine if the lander needs immediate angular stabilization to prevent loss of control.

- **Horizontal and Landing Gear Adjustments:**
  - **Condition:** X Coord Outside Wider Center, Right Leg Not in Contact
  - **Action:** Use the main engine when the lander is substantially off-center or the right leg is not contacting the ground to ensure the lander returns to a stable descent path.
