(define (stream panda-tamp)
  (:stream sample-pose
    :inputs (?o ?r)
    :domain (Stackable ?o ?r)
    :outputs (?p)
    :certified (and (Pose ?o ?p) (Supported ?o ?p ?r))
  )

  (:stream sample-grasp
    :inputs (?o)
    :domain (Graspable ?o)
    :outputs (?g)
    :certified (Grasp ?o ?g)
  )

  (:stream inverse-kinematics
    :inputs (?a ?o ?p ?g)
    :domain (and (Controllable ?a) (Pose ?o ?p) (Grasp ?o ?g))
    ; :outputs (?q ?t)
    :outputs (?t)
    ; :certified (and (AConf ?q) (ATraj ?t) (Kin ?a ?o ?p ?g ?q ?t)) ; (ATraj ?t)
    :certified (and (ATraj ?t) (Kin ?a ?o ?p ?g ?t));(Kin ?a ?o ?p ?g ?q ?t))
  )

  (:stream sample-loose-goal-push
    :inputs (?o)
    :domain (Graspable ?o)
    :outputs (?lg) ; out put a pose in the area
    :certified (and (LooseGoalPush ?o ?lg) (Pose ?o ?lg))
  )

  (:stream sample-loose-goal-hook
    :inputs (?o)
    :domain (Graspable ?o)
    :outputs (?lg) ; out put a pose in the area
    :certified (and (LooseGoalHook ?o ?lg) (Pose ?o ?lg))
  )

  (:stream sample-pushing
    :inputs (?a ?o ?p ?lg)
    :domain (and (Controllable ?a)(Graspable ?o)(Pose ?o ?p)(LooseGoal ?o ?lg))
    :outputs (?push) ; out put a pose in the area
    :certified (and (Pushing ?push))
  )

  (:stream sample-hooking
    :inputs (?a ?o ?p ?lg)
    :domain (and (Controllable ?a)(Graspable ?o)(HookInitial ?o ?p)(LooseGoal ?o ?lg))
    :outputs (?h) ; out put a pose in the area
    :certified (and (Hooking ?h))
  )

  ; (:stream sample-loose-goal
  ;   :input (?o ?p)
  ;   :domain (and (Pose ?o ?p))
  ;   :outputs (?p) ; out put a pose in the area
  ;   :certified (LooseGoal ?o ?p ?lg)
  ; )

  ; (:stream sample-push
  ;   :inputs (?a)
  ;   :domain (and (Pose ?o ?p) (Grasp ?o ?g))
  ;   :certified (CFreeApproachAny ?o ?p ?g ?o ?p)
  ; )

  ; (:stream test-cfree-pose-pose
  ;   :inputs (?o1 ?p1 ?o2 ?p2)
  ;   :domain (and (Pose ?o1 ?p1) (Pose ?o2 ?p2))
  ;   :certified (CFreePosePose ?o1 ?p1 ?o2 ?p2)
  ; )
  
  (:stream test-cfree-approach-pose
    :inputs (?o1 ?p1 ?g1 ?o2 ?p2)
    :domain (and (Pose ?o1 ?p1) (Grasp ?o1 ?g1) (Pose ?o2 ?p2))
    :certified (CFreeApproachPose ?o1 ?p1 ?g1 ?o2 ?p2)
  )

  (:stream test-push-initial-pose
    :inputs (?o ?p)
    :domain (and (Pose ?o ?p))
    :certified (PushInitial ?o ?p)
  )

  (:stream test-same-surface
    :inputs (?o ?p ?lg)
    :domain (and (Pose ?o ?p) (LooseGoal ?o ?lg))
    :certified (SameSurface ?p ?lg)
  )

  ; (:stream test-near-pose
  ;   :inputs (?o ?p)
  ;   :domain (and (Pose ?o ?p))
  ;   :certified (NearPose ?o ?p)
  ; )

  ; (:stream test-cfree-traj-pose
  ;   :inputs (?t ?o2 ?p2)
  ;   :domain (and (ATraj ?t) (Pose ?o2 ?p2))
  ;   :certified (CFreeTrajPose ?t ?o2 ?p2)
  ; )


  ;(:stream test-cfree-traj-grasp-pose
  ;  :inputs (?t ?a ?o1 ?g1 ?o2 ?p2)
  ;  :domain (and (BTraj ?t) (Arm ?a) (Grasp ?o1 ?g1) (Pose ?o2 ?p2))
  ;  :certified (CFreeTrajGraspPose ?t ?a ?o1 ?g1 ?o2 ?p2)
  ;)

  ; (:function (Distance ?q1 ?q2)
  ;   (and (BConf ?q1) (BConf ?q2))
  ; )
  ;(:function (MoveCost ?t)
  ;  (and (BTraj ?t))
  ;)
)