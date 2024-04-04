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
    :outputs (?t)
    :certified (and (ATraj ?t) (Kin ?a ?o ?p ?g ?t))
  )

  (:stream sample-loose-goal-push
    :inputs (?o)
    :domain (Graspable ?o)
    :outputs (?lg) ; out put a pose in the area
    :certified (and (LooseGoalPush ?o ?lg) (Pose ?o ?lg))
  )

  (:stream sample-loose-goal-retrieve
    :inputs (?o)
    :domain (Graspable ?o)
    :outputs (?lg) ; out put a pose in the area
    :certified (and (LooseGoalRetrieve ?o ?lg) (Pose ?o ?lg))
  )
  
  (:stream test-cfree-approach-pose
    :inputs (?o1 ?p1 ?g1 ?o2 ?p2)
    :domain (and (Pose ?o1 ?p1) (Grasp ?o1 ?g1) (Pose ?o2 ?p2))
    :certified (CFreeApproachPose ?o1 ?p1 ?g1 ?o2 ?p2)
  )

  (:stream test-push-initial-pose
    :inputs (?o ?p)
    :domain (and (Pose ?o ?p))
    :certified (and (PushInitial ?o ?p)(Pushable ?o))
  )

  (:stream test-retrieve-initial-pose
    :inputs (?o ?p)
    :domain (and (Pose ?o ?p))
    :certified (HookInitial ?o ?p)
  )

  ; (:stream test-same-surface
  ;   :inputs (?o ?p ?lg)
  ;   :domain (and (Pose ?o ?p) (LooseGoal ?o ?lg))
  ;   :certified (SameSurface ?p ?lg)
  ; )

)