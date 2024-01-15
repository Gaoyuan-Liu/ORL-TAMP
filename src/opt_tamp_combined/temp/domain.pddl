(define (domain panda-tamp)
  (:requirements :strips :equality)
  (:constants ); (:constants @sink @stove)
  (:predicates
    (Arm ?a)
    (Stackable ?o ?r)
    (Type ?t ?b)

    (Pose ?o ?p)
    (Grasp ?o ?g)
    (Kin ?a ?o ?p ?g ?t) ; comment ?t
    (BaseMotion ?q1 ?t ?q2)
    (ArmMotion ?a ?q1 ?t ?q2)
    (Supported ?o ?p ?r)
    ; (BTraj ?t) ; commented the line
    (ATraj ?t) ; commented the line
    (Lifted ?o)

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
  )
  (:functions
    (Distance ?q1 ?q2)
    (MoveCost ?t)
    (PickCost)
    (PlaceCost)
  )



  (:action pick
    :parameters (?a ?o ?p ?g ?q ?t) 
    :precondition (and ;(Solvable ?a)
                       (Kin ?a ?o ?p ?g ?t)
                       (AtPose ?o ?p) (HandEmpty ?a) 
                       (not (UnsafeApproach ?o ?p ?g))
                      ;  (not (UnsafeATraj ?t))
                  )
    :effect (and (AtGrasp ?a ?o ?g) (CanMove) 
                 (not (AtPose ?o ?p)) (not (HandEmpty ?a))
                 (Lifted ?o)
                 )
  )


  (:action place
    :parameters (?a ?o ?p ?g ?q ?t) 
    :precondition (and ;(Pose ?o ?p)
                      ;  (not (UnsafePose ?o ?p))
                       (Kin ?a ?o ?p ?g ?t) 
                       (AtGrasp ?a ?o ?g)
                       (Lifted ?o)
                      ; (not (UnsafeApproach ?o ?p ?g))
                      ;  (not (UnsafeATraj ?t))
                  )
    :effect (and (AtPose ?o ?p) (HandEmpty ?a) (CanMove)
                 (not (AtGrasp ?a ?o ?g))
                 )
  )

  (:action hold
    :parameters (?a ?o ?p ?g ?q ?t) 
    :precondition (and ;(Solvable ?a)
                       (Kin ?a ?o ?p ?g ?t)
                       (AtPose ?o ?p) (HandEmpty ?a) 
                       (not (UnsafeApproach ?o ?p ?g))
                      ;  (not (UnsafeATraj ?t))
                  )
    :effect (and (AtGrasp ?a ?o ?g) (CanMove)
                 (not (AtPose ?o ?p)) (not (HandEmpty ?a))
                 (not (Lifted ?o))
                 )
  )


  (:derived (On ?o ?r)
    (exists (?p) (and (Supported ?o ?p ?r)
                      (AtPose ?o ?p)))
  )

  (:derived (Holding ?a ?o)
    (exists (?g) (and (Arm ?a) (Grasp ?o ?g) (not (Lifted ?o))
                      (AtGrasp ?a ?o ?g)))
  )

  ; (:derived (UnsafePose ?o ?p)
  ;   (exists (?o2 ?p2) (and (Pose ?o ?p) (Pose ?o2 ?p2) (not (= ?o ?o2))
  ;                          (not (CFreePosePose ?o ?p ?o2 ?p2))
  ;                          (AtPose ?o2 ?p2)))
  ; )

  (:derived (UnsafeApproach ?o ?p ?g)
    (exists (?o2 ?p2) (and (Pose ?o ?p) (Grasp ?o ?g) (Pose ?o2 ?p2) (not (= ?o ?o2))
                           (not (CFreeApproachPose ?o ?p ?g ?o2 ?p2))
                           (AtPose ?o2 ?p2)))
  )

  ; (:derived (Solvable ?a)
  ;   (exists (?o ?p) (and (Pose ?o ?p) (Grasp ?o ?g) (CFreeApproachAny ?o ?p ?g)))
  ; )

  ; (:derived (UnsafeATraj ?t)
  ;   (exists (?o2 ?p2) (and (ATraj ?t) (Pose ?o2 ?p2)
  ;                          (not (CFreeTrajPose ?t ?o2 ?p2))
  ;                          (AtPose ?o2 ?p2)))
  ; )





 
)