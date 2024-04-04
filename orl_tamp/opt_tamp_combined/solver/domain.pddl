(define (domain panda-tamp)
  (:requirements :strips :equality)
  (:constants ); (:constants @sink @stove)
  (:predicates
    (Arm ?a)

    (Pose ?o ?p)
    (Grasp ?o ?g)
    (Kin ?a ?o ?p ?g ?t) 

    (Supported ?o ?p ?r)
    (LooseGoalPush ?o ?lg)
    (LooseGoalRetrieve ?o ?lg)
    (CFreeApproachPose ?o ?p ?g ?o2 ?p2)



    (AtPose ?o ?p)
    (AtGrasp ?a ?o ?g)
    (HandEmpty ?a)
    (CanMove)


    (On ?o ?r)


    (PushInitial ?o ?p)
    (HookInitial ?o ?p)
    (SameSurface ?p ?lg)

    (Around ?o ?p)
  )

  (:action pick
    :parameters (?a ?o ?p ?g ?t) 
    :precondition (and (Kin ?a ?o ?p ?g ?t)
                       (HandEmpty ?a) 
                       (AtPose ?o ?p) 
                  )
    :effect (and (AtGrasp ?a ?o ?g) 
                 (CanMove) 
                 (not (AtPose ?o ?p)) 
                 (not (HandEmpty ?a))
                 )
  )


  (:action place
    :parameters (?a ?o ?p ?g ?t) 
    :precondition (and ;(Pose ?o ?p)
                       (Kin ?a ?o ?p ?g ?t) 
                       (AtGrasp ?a ?o ?g)
                  
                  )
    :effect (and (AtPose ?o ?p) (HandEmpty ?a) (CanMove)
                 (not (AtGrasp ?a ?o ?g))
                 )
  )


  (:action retrieve
    :parameters (?a ?o ?p ?lg) 
    :precondition (and (Arm ?a) (Pose ?o ?p)
                       (HandEmpty ?a) 
                       (AtPose ?o ?p)
                       (LooseGoalRetrieve ?o ?lg)
                       (HookInitial ?o ?p)
                  )

    :effect (and (CanMove)
                 (not (AtPose ?o ?lg))
                 (Around ?o ?lg)
                 (not (AtPose ?o ?p))
                 )
  )

  (:action edgepush
    :parameters (?a ?o ?p ?lg) 
    :precondition (and (Arm ?a) (Pose ?o ?p)
                       (HandEmpty ?a) 
                       (AtPose ?o ?p)
                       (LooseGoalPush ?o ?lg)
                       (PushInitial ?o ?p)
                  )

    :effect (and (HandEmpty ?a) (CanMove)
                 (not (AtPose ?o ?lg))
                 (Around ?o ?lg)
                 (not (AtPose ?o ?p))
                 )
  )

  (:action observe_push
    :parameters (?a ?o ?lg)
    :precondition (and  
                        (Arm ?a) 
                        (LooseGoalPush ?o ?lg)
                        (Around ?o ?lg)
                   )
    :effect (and  
                  (HandEmpty ?a) (CanMove) 
                  (AtPose ?o ?lg)
                  )
    )

  (:action observe_retrieve
    :parameters (?a ?o ?lg)
    :precondition (and 
                        (Arm ?a) 
                        (LooseGoalRetrieve ?o ?lg)
                        (Around ?o ?lg)
                   )
    :effect (and  
                  (HandEmpty ?a) (CanMove) 
                  (AtPose ?o ?lg)
                  )
    )
    


  (:derived (On ?o ?r)
    (exists (?p) (and (Supported ?o ?p ?r)
                      (AtPose ?o ?p)))
  )


 
)