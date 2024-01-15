(define (domain panda-tamp)
  (:requirements :strips :equality)
  (:constants ); (:constants @sink @stove)
  (:predicates
    (Arm ?a)
    (Stackable ?o ?r)
    (Type ?t ?b)

    (Pose ?o ?p)
    (Grasp ?o ?g)
    (Kin ?a ?o ?p ?g ?t) 
    (BaseMotion ?q1 ?t ?q2)
    (ArmMotion ?a ?q1 ?t ?q2)
    (Supported ?o ?p ?r)
    (ATraj ?t) 
    (LooseGoal ?o ?lg)

    (CFreePosePose ?o ?p ?o2 ?p2)
    (CFreeApproachPose ?o ?p ?g ?o2 ?p2)
    ; (CFreeTrajPose ?t ?o2 ?p2)
    ; (CFreeTrajGraspPose ?t ?a ?o1 ?g1 ?o2 ?p2)
    (Solvable ?a)

    (AtPose ?o ?p)
    (AtGrasp ?a ?o ?g)
    (HandEmpty ?a)
    ; (AtBConf ?q)
    (AtAConf ?a ?q)
    (CanMove)
    (Cleaned ?o)
    (Cooked ?o)

    (On ?o ?r)
    ; (On2 ?o1 ?o2 ?r)
    ; (On3 ?o1 ?o2 ?o3 ?r)
    ; (On4 ?o1 ?o2 ?o3 ?o4 ?r)
    ; (On5 ?o1 ?o2 ?o3 ?o4 ?o5 ?r)

    (Holding ?a ?o)
    (UnsafePose ?o ?p)
    (UnsafeApproach ?o ?p ?g)
    (UnsafeATraj ?t)
    (UnsafeBTraj ?t)
    (HookInitial ?o ?p)
    (NearPose ?o ?p)

    (Around ?o ?p)

  )
  (:functions
    (Distance ?q1 ?q2)
    (MoveCost ?t)
    (PickCost)
    (PlaceCost)
  )



  (:action pick
    :parameters (?a ?o ?p ?g ?q ?t) 
    :precondition (and (Kin ?a ?o ?p ?g ?t)
                       (AtPose ?o ?p) (HandEmpty ?a) 
                       (NearPose ?o ?p)
                       (not (UnsafeApproach ?o ?p ?g))

                      ;  (not (UnsafeATraj ?t))
                  )
    :effect (and (AtGrasp ?a ?o ?g) (CanMove) 
                 (not (AtPose ?o ?p)) (not (HandEmpty ?a))
                 )
  )


  (:action place
    :parameters (?a ?o ?p ?g ?q ?t) 
    :precondition (and ;(Pose ?o ?p)
                      ;  (not (UnsafePose ?o ?p))
                       (Kin ?a ?o ?p ?g ?t) 
                       (AtGrasp ?a ?o ?g)
                      ; (not (UnsafeApproach ?o ?p ?g))
                      ;  (not (UnsafeATraj ?t))
                  )
    :effect (and (AtPose ?o ?p) (HandEmpty ?a) (CanMove)
                 (not (AtGrasp ?a ?o ?g))
                 )
  )


  ; (:action hold
  ;   :parameters (?a ?o ?p ?g ?q ?t) 
  ;   :precondition (and ;(Solvable ?a)
  ;                      (Kin ?a ?o ?p ?g ?t)
  ;                      (AtPose ?o ?p) (HandEmpty ?a) 
  ;                      (not (UnsafeApproach ?o ?p ?g))
  ;                     ;  (not (UnsafeATraj ?t))
  ;                 )
  ;   :effect (and (AtGrasp ?a ?o ?g) (CanMove)
  ;                (not (AtPose ?o ?p)) (not (HandEmpty ?a))
  ;                (not (Lifted ?o))
  ;                )
  ; )

  ; (:action hook
  ;   :parameters (?a ?o ?p ?lg) ; arm, object, initial_pose, end_pose2
  ;   :precondition (and (Arm ?a) (Pose ?o ?p)
  ;                      (HandEmpty ?a) 
  ;                      (AtPose ?o ?p)
  ;                      (LooseGoal ?o ?lg)
  ;                      (HookInitial ?o ?p)
  ;                      ; here need to distinguish the difference of hook and pick-place
  ;                 )

  ;   :effect (and (HandEmpty ?a) (CanMove)
  ;                ;(not (AtGrasp ?a ?o ?g))
  ;                (Around ?o ?lg)
  ;                (not (AtPose ?o ?p))
  ;                )
  ; )

  ; (:action observe
  ;   :parameters (?a ?o ?lg)
  ;   :precondition (and
  ;                      (Around ?o ?lg)
  ;                  )
  ;   :effect (and  
  ;                 (HandEmpty ?a) (CanMove) 
  ;                 (AtPose ?o ?lg)
  ;                 ; (not (AtGrasp ?a ?o ?g))
  ;                 ;(not (AtGrasp ?a ?o ?g))
  ;                 )
  ;   )
    


  (:derived (On ?o ?r)
    (exists (?p) (and (Supported ?o ?p ?r)
                      (AtPose ?o ?p)))
  )

  ; (:derived (Holding ?a ?o)
  ;   (exists (?g) (and (Arm ?a) (Grasp ?o ?g) (not (Lifted ?o))
  ;                     (AtGrasp ?a ?o ?g)))
  ; )


  (:derived (UnsafeApproach ?o ?p ?g)
    (exists (?o2 ?p2) (and (Pose ?o ?p) (Grasp ?o ?g) (Pose ?o2 ?p2) (not (= ?o ?o2))
                           (not (CFreeApproachPose ?o ?p ?g ?o2 ?p2))
                           (AtPose ?o2 ?p2)))
  )




  ; (:derived (UnsafePose ?o ?p)
  ;   (exists (?o2 ?p2) (and (Pose ?o ?p) (Pose ?o2 ?p2) (not (= ?o ?o2))
  ;                          (not (CFreePosePose ?o ?p ?o2 ?p2))
  ;                          (AtPose ?o2 ?p2)))
  ; )


  ; (:derived (Solvable ?a)
  ;   (exists (?o ?p) (and (Pose ?o ?p) (Grasp ?o ?g) (CFreeApproachAny ?o ?p ?g)))
  ; )

  ; (:derived (UnsafeATraj ?t)
  ;   (exists (?o2 ?p2) (and (ATraj ?t) (Pose ?o2 ?p2)
  ;                          (not (CFreeTrajPose ?t ?o2 ?p2))
  ;                          (AtPose ?o2 ?p2)))
  ; )





 
)